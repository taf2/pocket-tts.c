#include "ptts_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *host;
    float *device;
    size_t bytes;
} ptts_weight_cache_entry;

static ptts_weight_cache_entry *g_cache = NULL;
static int g_cache_len = 0;
static int g_cache_cap = 0;
static cublasHandle_t g_handle;
static int g_inited = 0;
static float *g_tmp_x = NULL;
static float *g_tmp_y = NULL;
static size_t g_tmp_x_bytes = 0;
static size_t g_tmp_y_bytes = 0;
static CUmodule g_mod;
static CUfunction g_conv1d;
static CUfunction g_convtr1d;
static CUfunction g_elu;
static CUfunction g_add;
static int g_kernels_ready = 0;
static float *g_conv_buf0 = NULL;
static float *g_conv_buf1 = NULL;
static float *g_conv_buf2 = NULL;
static size_t g_conv_buf0_bytes = 0;
static size_t g_conv_buf1_bytes = 0;
static size_t g_conv_buf2_bytes = 0;

static int ensure_kernels(void) {
    if (g_kernels_ready) return 0;
    if (cuInit(0) != CUDA_SUCCESS) return -1;

    int dev = 0;
    struct cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return -1;
    char arch[64];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", prop.major, prop.minor);

    const char *src =
        "extern \"C\" __global__ void conv1d_kernel(const float* x, const float* w, const float* b, float* y, int in_ch, int out_ch, int T, int k, int stride, int groups, int out_len, int left_pad) {\\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\\n"
        "  int total = out_ch * out_len;\\n"
        "  if (idx >= total) return;\\n"
        "  int oc = idx / out_len;\\n"
        "  int t = idx - oc * out_len;\\n"
        "  int out_per_group = out_ch / groups;\\n"
        "  int in_per_group = in_ch / groups;\\n"
        "  int g = oc / out_per_group;\\n"
        "  int in_base = g * in_per_group;\\n"
        "  const float* wbase = w + (size_t)oc * in_per_group * k;\\n"
        "  float sum = b ? b[oc] : 0.0f;\\n"
        "  int in_start = t * stride - left_pad;\\n"
        "  for (int ic = 0; ic < in_per_group; ic++) {\\n"
        "    const float* wrow = wbase + ic * k;\\n"
        "    const float* xch = x + (size_t)(in_base + ic) * T;\\n"
        "    for (int kk = 0; kk < k; kk++) {\\n"
        "      int xi = in_start + kk;\\n"
        "      if (xi < 0 || xi >= T) continue;\\n"
        "      sum += wrow[kk] * xch[xi];\\n"
        "    }\\n"
        "  }\\n"
        "  y[(size_t)oc * out_len + t] = sum;\\n"
        "}\\n"
        "extern \"C\" __global__ void convtr1d_kernel(const float* x, const float* w, const float* b, float* y, int in_ch, int out_ch, int T, int k, int stride, int groups, int out_len) {\\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\\n"
        "  int total = out_ch * out_len;\\n"
        "  if (idx >= total) return;\\n"
        "  int oc = idx / out_len;\\n"
        "  int t = idx - oc * out_len;\\n"
        "  int out_per_group = out_ch / groups;\\n"
        "  int in_per_group = in_ch / groups;\\n"
        "  int g = oc / out_per_group;\\n"
        "  int in_base = g * in_per_group;\\n"
        "  float sum = b ? b[oc] : 0.0f;\\n"
        "  for (int ic = 0; ic < in_per_group; ic++) {\\n"
        "    const float* xch = x + (size_t)(in_base + ic) * T;\\n"
        "    const float* wbase = w + (size_t)(in_base + ic) * out_per_group * k;\\n"
        "    for (int kk = 0; kk < k; kk++) {\\n"
        "      int ti = t - kk;\\n"
        "      if (ti < 0) continue;\\n"
        "      if (ti % stride) continue;\\n"
        "      int xi = ti / stride;\\n"
        "      if (xi < 0 || xi >= T) continue;\\n"
        "      sum += wbase[(oc - g * out_per_group) * k + kk] * xch[xi];\\n"
        "    }\\n"
        "  }\\n"
        "  y[(size_t)oc * out_len + t] = sum;\\n"
        "}\\n"
        "extern \"C\" __global__ void elu_kernel(float* x, int n) {\\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\\n"
        "  if (i >= n) return;\\n"
        "  float v = x[i];\\n"
        "  x[i] = v >= 0.0f ? v : expf(v) - 1.0f;\\n"
        "}\\n"
        "extern \"C\" __global__ void add_kernel(float* a, const float* b, int n) {\\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\\n"
        "  if (i >= n) return;\\n"
        "  a[i] += b[i];\\n"
        "}\\n";

    nvrtcProgram prog;
    nvrtcResult rc = nvrtcCreateProgram(&prog, src, "ptts_kernels.cu", 0, NULL, NULL);
    if (rc != NVRTC_SUCCESS) return -1;
    const char *opts[] = {arch};
    rc = nvrtcCompileProgram(prog, 1, opts);
    if (rc != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        return -1;
    }
    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    char *ptx = (char *)malloc(ptx_size);
    if (!ptx) {
        nvrtcDestroyProgram(&prog);
        return -1;
    }
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    if (cuModuleLoadData(&g_mod, ptx) != CUDA_SUCCESS) {
        free(ptx);
        return -1;
    }
    free(ptx);
    if (cuModuleGetFunction(&g_conv1d, g_mod, "conv1d_kernel") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&g_convtr1d, g_mod, "convtr1d_kernel") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&g_elu, g_mod, "elu_kernel") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&g_add, g_mod, "add_kernel") != CUDA_SUCCESS) return -1;

    g_kernels_ready = 1;
    return 0;
}

static int ensure_device_buffer(float **buf, size_t *cap, size_t bytes) {
    if (*cap >= bytes && *buf) return 0;
    if (*buf) {
        cudaFree(*buf);
        *buf = NULL;
        *cap = 0;
    }
    if (cudaMalloc((void **)buf, bytes) != cudaSuccess) return -1;
    *cap = bytes;
    return 0;
}

static int ensure_conv_buffer(float **buf, size_t *cap, size_t bytes) {
    return ensure_device_buffer(buf, cap, bytes);
}

static int ensure_init(void) {
    if (g_inited) return 0;
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    g_inited = 1;
    atexit(ptts_cuda_shutdown);
    return 0;
}

static float *get_weight_device(const float *host, size_t bytes) {
    for (int i = 0; i < g_cache_len; i++) {
        if (g_cache[i].host == host) return g_cache[i].device;
    }
    float *dev = NULL;
    if (cudaMalloc((void **)&dev, bytes) != cudaSuccess) return NULL;
    if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        return NULL;
    }
    if (g_cache_len == g_cache_cap) {
        int next_cap = g_cache_cap ? g_cache_cap * 2 : 32;
        ptts_weight_cache_entry *next = (ptts_weight_cache_entry *)realloc(
            g_cache, (size_t)next_cap * sizeof(*g_cache));
        if (!next) {
            cudaFree(dev);
            return NULL;
        }
        g_cache = next;
        g_cache_cap = next_cap;
    }
    g_cache[g_cache_len].host = host;
    g_cache[g_cache_len].device = dev;
    g_cache[g_cache_len].bytes = bytes;
    g_cache_len++;
    return dev;
}

int ptts_cuda_linear_forward(float *y, const float *x, const float *w, const float *b,
                             int n, int in, int out) {
    if (ensure_init() != 0) return -1;

    size_t x_bytes = (size_t)n * (size_t)in * sizeof(float);
    size_t y_bytes = (size_t)n * (size_t)out * sizeof(float);
    size_t w_bytes = (size_t)out * (size_t)in * sizeof(float);

    float *d_w = get_weight_device(w, w_bytes);
    if (!d_w) return -1;

    if (ensure_device_buffer(&g_tmp_x, &g_tmp_x_bytes, x_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_tmp_y, &g_tmp_y_bytes, y_bytes) != 0) return -1;
    if (cudaMemcpy(g_tmp_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* Column-major: C(out x n) = W^T(out x in) * X(in x n).
     * Memory of C matches row-major [n x out] directly. */
    cublasStatus_t st = cublasSgemm(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out, n, in,
        &alpha,
        d_w, in,
        g_tmp_x, in,
        &beta,
        g_tmp_y, out);

    if (st != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    if (cudaMemcpy(y, g_tmp_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;

    if (b) {
        for (int t = 0; t < n; t++) {
            float *yrow = y + (size_t)t * out;
            for (int o = 0; o < out; o++) yrow[o] += b[o];
        }
    }
    return 0;
}

int ptts_cuda_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                             int in_ch, int out_ch, int T, int k, int stride, int groups) {
    if (ensure_kernels() != 0) return -1;
    int out_len = T / stride;
    int left_pad = k - stride;
    size_t x_bytes = (size_t)in_ch * T * sizeof(float);
    size_t y_bytes = (size_t)out_ch * out_len * sizeof(float);
    size_t w_bytes = (size_t)out_ch * (in_ch / groups) * k * sizeof(float);

    if (ensure_device_buffer(&g_tmp_x, &g_tmp_x_bytes, x_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_tmp_y, &g_tmp_y_bytes, y_bytes) != 0) return -1;
    if (cudaMemcpy(g_tmp_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    float *d_w = get_weight_device(w, w_bytes);
    float *d_b = NULL;
    if (b) {
        d_b = get_weight_device(b, (size_t)out_ch * sizeof(float));
        if (!d_b) return -1;
    }

    int total = out_ch * out_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    void *args[] = {&g_tmp_x, &d_w, &d_b, &g_tmp_y, &in_ch, &out_ch, &T, &k,
                    &stride, &groups, &out_len, &left_pad};
    if (cuLaunchKernel(g_conv1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return -1;
    }
    if (cudaMemcpy(y, g_tmp_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

int ptts_cuda_convtr1d_forward(float *y, const float *x, const float *w, const float *b,
                               int in_ch, int out_ch, int T, int k, int stride, int groups) {
    if (ensure_kernels() != 0) return -1;
    int out_len = T * stride;
    size_t x_bytes = (size_t)in_ch * T * sizeof(float);
    size_t y_bytes = (size_t)out_ch * out_len * sizeof(float);
    size_t w_bytes = (size_t)in_ch * (out_ch / groups) * k * sizeof(float);

    if (ensure_device_buffer(&g_tmp_x, &g_tmp_x_bytes, x_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_tmp_y, &g_tmp_y_bytes, y_bytes) != 0) return -1;
    if (cudaMemcpy(g_tmp_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    float *d_w = get_weight_device(w, w_bytes);
    float *d_b = NULL;
    if (b) {
        d_b = get_weight_device(b, (size_t)out_ch * sizeof(float));
        if (!d_b) return -1;
    }

    int total = out_ch * out_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    void *args[] = {&g_tmp_x, &d_w, &d_b, &g_tmp_y, &in_ch, &out_ch, &T, &k,
                    &stride, &groups, &out_len};
    if (cuLaunchKernel(g_convtr1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return -1;
    }
    if (cudaMemcpy(y, g_tmp_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

static int conv1d_device(float *y, const float *x, const float *w, const float *b,
                         int in_ch, int out_ch, int T, int k, int stride, int groups) {
    if (ensure_kernels() != 0) return -1;
    int out_len = T / stride;
    int left_pad = k - stride;
    int total = out_ch * out_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    void *args[] = {&x, &w, &b, &y, &in_ch, &out_ch, &T, &k,
                    &stride, &groups, &out_len, &left_pad};
    if (cuLaunchKernel(g_conv1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return -1;
    }
    return 0;
}

static int convtr1d_device(float *y, const float *x, const float *w, const float *b,
                           int in_ch, int out_ch, int T, int k, int stride, int groups) {
    if (ensure_kernels() != 0) return -1;
    int full_len = (T - 1) * stride + k;
    int out_len = full_len - (k - stride);
    int total = out_ch * out_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    void *args[] = {&x, &w, &b, &y, &in_ch, &out_ch, &T, &k,
                    &stride, &groups, &out_len};
    if (cuLaunchKernel(g_convtr1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return -1;
    }
    return 0;
}

static int elu_device(float *x, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&x, &n};
    if (cuLaunchKernel(g_elu, grid, 1, 1, block, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return -1;
    }
    return 0;
}

static int add_device(float *a, const float *b, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&a, &b, &n};
    if (cuLaunchKernel(g_add, grid, 1, 1, block, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        return -1;
    }
    return 0;
}

int ptts_cuda_mimi_convstack(const ptts_cuda_conv1d_desc *dec_in,
                             const ptts_cuda_convtr1d_desc *up0,
                             const ptts_cuda_conv1d_desc *res0_1,
                             const ptts_cuda_conv1d_desc *res0_2,
                             const ptts_cuda_convtr1d_desc *up1,
                             const ptts_cuda_conv1d_desc *res1_1,
                             const ptts_cuda_conv1d_desc *res1_2,
                             const ptts_cuda_convtr1d_desc *up2,
                             const ptts_cuda_conv1d_desc *res2_1,
                             const ptts_cuda_conv1d_desc *res2_2,
                             const ptts_cuda_conv1d_desc *dec_out,
                             const float *x_host, int T,
                             float *out_host, int *out_len) {
    if (!dec_in || !up0 || !res0_1 || !res0_2 || !up1 || !res1_1 || !res1_2 ||
        !up2 || !res2_1 || !res2_2 || !dec_out || !x_host || !out_host || !out_len) {
        return -1;
    }
    if (ensure_kernels() != 0) return -1;

    size_t x_bytes = (size_t)dec_in->in_ch * T * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, x_bytes) != 0) return -1;
    float *d_x = g_conv_buf0;
    if (cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }

    int t = T;
    int out_t = t / dec_in->stride;
    size_t y_bytes = (size_t)dec_in->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, y_bytes) != 0) return -1;
    float *d_y = g_conv_buf1;
    float *d_w = get_weight_device(dec_in->w, (size_t)dec_in->out_ch * (dec_in->in_ch / dec_in->groups) * dec_in->k * sizeof(float));
    float *d_b = dec_in->b ? get_weight_device(dec_in->b, (size_t)dec_in->out_ch * sizeof(float)) : NULL;
    if (!d_w) return -1;
    if (conv1d_device(d_y, d_x, d_w, d_b, dec_in->in_ch, dec_in->out_ch, t, dec_in->k, dec_in->stride, dec_in->groups) != 0) {
        return -1;
    }
    d_x = d_y;
    t = out_t;

    if (elu_device(d_x, dec_in->out_ch * t) != 0) return -1;

    out_t = t * up0->stride;
    y_bytes = (size_t)up0->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf0;
    d_w = get_weight_device(up0->w, (size_t)up0->in_ch * (up0->out_ch / up0->groups) * up0->k * sizeof(float));
    d_b = up0->b ? get_weight_device(up0->b, (size_t)up0->out_ch * sizeof(float)) : NULL;
    if (!d_w) return -1;
    if (convtr1d_device(d_y, d_x, d_w, d_b, up0->in_ch, up0->out_ch, t, up0->k, up0->stride, up0->groups) != 0) {
        return -1;
    }
    d_x = d_y;
    t = out_t;

    float *d_tmp1 = NULL;
    float *d_tmp2 = NULL;
    size_t tmp_bytes = (size_t)res0_1->out_ch * t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, tmp_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, tmp_bytes) != 0) return -1;
    d_tmp1 = g_conv_buf0;
    d_tmp2 = g_conv_buf2;
    if (cudaMemcpy(d_tmp1, d_x, tmp_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return -1;
    }
    if (elu_device(d_tmp1, res0_1->out_ch * t) != 0) return -1;
    d_w = get_weight_device(res0_1->w, (size_t)res0_1->out_ch * (res0_1->in_ch / res0_1->groups) * res0_1->k * sizeof(float));
    d_b = res0_1->b ? get_weight_device(res0_1->b, (size_t)res0_1->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_tmp2, d_tmp1, d_w, d_b, res0_1->in_ch, res0_1->out_ch, t, res0_1->k, res0_1->stride, res0_1->groups) != 0) {
        return -1;
    }
    if (elu_device(d_tmp2, res0_1->out_ch * t) != 0) return -1;
    d_w = get_weight_device(res0_2->w, (size_t)res0_2->out_ch * (res0_2->in_ch / res0_2->groups) * res0_2->k * sizeof(float));
    d_b = res0_2->b ? get_weight_device(res0_2->b, (size_t)res0_2->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_tmp1, d_tmp2, d_w, d_b, res0_2->in_ch, res0_2->out_ch, t, res0_2->k, res0_2->stride, res0_2->groups) != 0) {
        return -1;
    }
    if (add_device(d_x, d_tmp1, res0_2->out_ch * t) != 0) return -1;

    if (elu_device(d_x, res0_2->out_ch * t) != 0) return -1;

    out_t = t * up1->stride;
    y_bytes = (size_t)up1->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf1;
    d_w = get_weight_device(up1->w, (size_t)up1->in_ch * (up1->out_ch / up1->groups) * up1->k * sizeof(float));
    d_b = up1->b ? get_weight_device(up1->b, (size_t)up1->out_ch * sizeof(float)) : NULL;
    if (convtr1d_device(d_y, d_x, d_w, d_b, up1->in_ch, up1->out_ch, t, up1->k, up1->stride, up1->groups) != 0) {
        return -1;
    }
    d_x = d_y;
    t = out_t;

    tmp_bytes = (size_t)res1_1->out_ch * t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, tmp_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, tmp_bytes) != 0) return -1;
    d_tmp1 = g_conv_buf0;
    d_tmp2 = g_conv_buf2;
    if (cudaMemcpy(d_tmp1, d_x, tmp_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return -1;
    }
    if (elu_device(d_tmp1, res1_1->out_ch * t) != 0) return -1;
    d_w = get_weight_device(res1_1->w, (size_t)res1_1->out_ch * (res1_1->in_ch / res1_1->groups) * res1_1->k * sizeof(float));
    d_b = res1_1->b ? get_weight_device(res1_1->b, (size_t)res1_1->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_tmp2, d_tmp1, d_w, d_b, res1_1->in_ch, res1_1->out_ch, t, res1_1->k, res1_1->stride, res1_1->groups) != 0) {
        return -1;
    }
    if (elu_device(d_tmp2, res1_1->out_ch * t) != 0) return -1;
    d_w = get_weight_device(res1_2->w, (size_t)res1_2->out_ch * (res1_2->in_ch / res1_2->groups) * res1_2->k * sizeof(float));
    d_b = res1_2->b ? get_weight_device(res1_2->b, (size_t)res1_2->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_tmp1, d_tmp2, d_w, d_b, res1_2->in_ch, res1_2->out_ch, t, res1_2->k, res1_2->stride, res1_2->groups) != 0) {
        return -1;
    }
    if (add_device(d_x, d_tmp1, res1_2->out_ch * t) != 0) return -1;

    if (elu_device(d_x, res1_2->out_ch * t) != 0) return -1;

    out_t = t * up2->stride;
    y_bytes = (size_t)up2->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf0;
    d_w = get_weight_device(up2->w, (size_t)up2->in_ch * (up2->out_ch / up2->groups) * up2->k * sizeof(float));
    d_b = up2->b ? get_weight_device(up2->b, (size_t)up2->out_ch * sizeof(float)) : NULL;
    if (convtr1d_device(d_y, d_x, d_w, d_b, up2->in_ch, up2->out_ch, t, up2->k, up2->stride, up2->groups) != 0) {
        return -1;
    }
    d_x = d_y;
    t = out_t;

    tmp_bytes = (size_t)res2_1->out_ch * t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, tmp_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, tmp_bytes) != 0) return -1;
    d_tmp1 = g_conv_buf1;
    d_tmp2 = g_conv_buf2;
    if (cudaMemcpy(d_tmp1, d_x, tmp_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return -1;
    }
    if (elu_device(d_tmp1, res2_1->out_ch * t) != 0) return -1;
    d_w = get_weight_device(res2_1->w, (size_t)res2_1->out_ch * (res2_1->in_ch / res2_1->groups) * res2_1->k * sizeof(float));
    d_b = res2_1->b ? get_weight_device(res2_1->b, (size_t)res2_1->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_tmp2, d_tmp1, d_w, d_b, res2_1->in_ch, res2_1->out_ch, t, res2_1->k, res2_1->stride, res2_1->groups) != 0) {
        return -1;
    }
    if (elu_device(d_tmp2, res2_1->out_ch * t) != 0) return -1;
    d_w = get_weight_device(res2_2->w, (size_t)res2_2->out_ch * (res2_2->in_ch / res2_2->groups) * res2_2->k * sizeof(float));
    d_b = res2_2->b ? get_weight_device(res2_2->b, (size_t)res2_2->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_tmp1, d_tmp2, d_w, d_b, res2_2->in_ch, res2_2->out_ch, t, res2_2->k, res2_2->stride, res2_2->groups) != 0) {
        return -1;
    }
    if (add_device(d_x, d_tmp1, res2_2->out_ch * t) != 0) return -1;

    if (elu_device(d_x, res2_2->out_ch * t) != 0) return -1;

    out_t = t / dec_out->stride;
    y_bytes = (size_t)dec_out->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf1;
    d_w = get_weight_device(dec_out->w, (size_t)dec_out->out_ch * (dec_out->in_ch / dec_out->groups) * dec_out->k * sizeof(float));
    d_b = dec_out->b ? get_weight_device(dec_out->b, (size_t)dec_out->out_ch * sizeof(float)) : NULL;
    if (conv1d_device(d_y, d_x, d_w, d_b, dec_out->in_ch, dec_out->out_ch, t, dec_out->k, dec_out->stride, dec_out->groups) != 0) {
        return -1;
    }

    if (cudaMemcpy(out_host, d_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -1;
    }
    *out_len = out_t;
    return 0;
}

void ptts_cuda_shutdown(void) {
    if (!g_inited) return;
    for (int i = 0; i < g_cache_len; i++) {
        cudaFree(g_cache[i].device);
    }
    free(g_cache);
    g_cache = NULL;
    g_cache_len = 0;
    g_cache_cap = 0;
    if (g_tmp_x) cudaFree(g_tmp_x);
    if (g_tmp_y) cudaFree(g_tmp_y);
    g_tmp_x = NULL;
    g_tmp_y = NULL;
    g_tmp_x_bytes = 0;
    g_tmp_y_bytes = 0;
    if (g_kernels_ready) {
        cuModuleUnload(g_mod);
        g_kernels_ready = 0;
    }
    if (g_conv_buf0) cudaFree(g_conv_buf0);
    if (g_conv_buf1) cudaFree(g_conv_buf1);
    if (g_conv_buf2) cudaFree(g_conv_buf2);
    g_conv_buf0 = NULL;
    g_conv_buf1 = NULL;
    g_conv_buf2 = NULL;
    g_conv_buf0_bytes = 0;
    g_conv_buf1_bytes = 0;
    g_conv_buf2_bytes = 0;
    cublasDestroy(g_handle);
    g_inited = 0;
}
