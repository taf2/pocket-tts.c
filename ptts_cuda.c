#include "ptts_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptts_kernels.h"

#define FLOWLM_FLOW_DIM 512
#define FLOWLM_LATENT_DIM 32
#define FLOWLM_D_MODEL 1024
#define FLOWLM_NUM_LAYERS 6
#define FLOWLM_NUM_HEADS 16
#define FLOWLM_HEAD_DIM 64

static void cuda_log_error(const char *where, cudaError_t err) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "[ptts] CUDA error at %s: %s\n", where, cudaGetErrorString(err));
}

static void cu_log_error(const char *where, CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *msg = NULL;
    cuGetErrorString(err, &msg);
    fprintf(stderr, "[ptts] CUDA driver error at %s: %s\n", where, msg ? msg : "unknown");
}

static int g_cuda_debug_inited = 0;
static int g_cuda_debug = 0;
static int g_cuda_convtr_inited = 0;
static int g_cuda_convtr_enabled = 1;
static int g_cuda_conv1d_inited = 0;
static int g_cuda_conv1d_enabled = 1;
#ifdef PTTS_CUDA_VALIDATE
static int g_cuda_validate_inited = 0;
static int g_cuda_validate_enabled = 0;
#endif
static int g_flow_profile_inited = 0;
static int g_flow_profile_enabled = 0;

static int cuda_debug_enabled(void) {
    if (!g_cuda_debug_inited) {
        const char *v = getenv("PTTS_CUDA_DEBUG");
        g_cuda_debug = (v && v[0] && strcmp(v, "0") != 0);
        g_cuda_debug_inited = 1;
    }
    return g_cuda_debug;
}

static int cuda_convtr_enabled(void) {
    if (!g_cuda_convtr_inited) {
        const char *v = getenv("PTTS_CUDA_CONVTR");
        g_cuda_convtr_enabled = !(v && v[0] && strcmp(v, "0") == 0);
        g_cuda_convtr_inited = 1;
    }
    return g_cuda_convtr_enabled;
}

static int cuda_conv1d_enabled(void) {
    if (!g_cuda_conv1d_inited) {
        const char *v = getenv("PTTS_CUDA_CONV1D");
        g_cuda_conv1d_enabled = !(v && v[0] && strcmp(v, "0") == 0);
        g_cuda_conv1d_inited = 1;
    }
    return g_cuda_conv1d_enabled;
}

#ifdef PTTS_CUDA_VALIDATE
static int cuda_validate_enabled(void) {
    if (!g_cuda_validate_inited) {
        const char *v = getenv("PTTS_CUDA_VALIDATE");
        g_cuda_validate_enabled = (v && v[0] && strcmp(v, "0") != 0);
        g_cuda_validate_inited = 1;
    }
    return g_cuda_validate_enabled;
}
#else
static int cuda_validate_enabled(void) {
    return 0;
}
#endif

static int flow_profile_enabled(void) {
    if (!g_flow_profile_inited) {
        const char *v = getenv("PTTS_FLOWNET_PROFILE");
        g_flow_profile_enabled = (v && v[0] && strcmp(v, "0") != 0);
        g_flow_profile_inited = 1;
    }
    return g_flow_profile_enabled;
}

static int cuda_check(const char *where) {
    if (!cuda_debug_enabled()) return 0;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cuda_log_error(where, err);
        return -1;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cuda_log_error(where, err);
        return -1;
    }
    return 0;
}

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
static CUfunction g_silu;
static CUfunction g_add3;
static CUfunction g_affine;
static CUfunction g_gate_add;
static CUfunction g_bias;
static CUfunction g_layernorm;
static CUfunction g_rmsnorm;
static CUfunction g_attn_scores;
static CUfunction g_attn_softmax;
static CUfunction g_attn_apply;
static CUfunction g_attn_fused;
static CUfunction g_attn_step;
static int g_kernels_ready = 0;
static float *g_conv_buf0 = NULL;
static float *g_conv_buf1 = NULL;
static float *g_conv_buf2 = NULL;
static float *g_conv_buf3 = NULL;
static size_t g_conv_buf0_bytes = 0;
static size_t g_conv_buf1_bytes = 0;
static size_t g_conv_buf2_bytes = 0;
static size_t g_conv_buf3_bytes = 0;
static float *g_flow_buf0 = NULL;
static float *g_flow_buf1 = NULL;
static float *g_flow_buf2 = NULL;
static float *g_flow_buf3 = NULL;
static float *g_flow_buf4 = NULL;
static float *g_flow_buf5 = NULL;
static float *g_flow_buf6 = NULL;
static float *g_flow_buf7 = NULL;
static float *g_flow_buf8 = NULL;
static float *g_flow_buf9 = NULL;
static float *g_flow_buf10 = NULL;
static size_t g_flow_buf0_bytes = 0;
static size_t g_flow_buf1_bytes = 0;
static size_t g_flow_buf2_bytes = 0;
static size_t g_flow_buf3_bytes = 0;
static size_t g_flow_buf4_bytes = 0;
static size_t g_flow_buf5_bytes = 0;
static size_t g_flow_buf6_bytes = 0;
static size_t g_flow_buf7_bytes = 0;
static size_t g_flow_buf8_bytes = 0;
static size_t g_flow_buf9_bytes = 0;
static size_t g_flow_buf10_bytes = 0;
static float *g_attn_q = NULL;
static float *g_attn_k = NULL;
static float *g_attn_v = NULL;
static float *g_attn_scores_buf = NULL;
static float *g_attn_out = NULL;
static size_t g_attn_q_bytes = 0;
static size_t g_attn_k_bytes = 0;
static size_t g_attn_v_bytes = 0;
static size_t g_attn_scores_bytes = 0;
static size_t g_attn_out_bytes = 0;
static float *g_k_cache_dev[FLOWLM_NUM_LAYERS];
static float *g_v_cache_dev[FLOWLM_NUM_LAYERS];
static size_t g_kv_cache_bytes = 0;
static int g_kv_cache_max_len = 0;

static int ensure_kernels(void) {
    if (g_kernels_ready) return 0;
    CUresult cuerr = cuInit(0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuInit", cuerr);
        return -1;
    }
    CUcontext ctx = NULL;
    cuerr = cuCtxGetCurrent(&ctx);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuCtxGetCurrent", cuerr);
        return -1;
    }
    if (!ctx) {
        CUdevice cu_dev;
        cuerr = cuDeviceGet(&cu_dev, 0);
        if (cuerr != CUDA_SUCCESS) {
            cu_log_error("cuDeviceGet", cuerr);
            return -1;
        }
        cuerr = cuCtxCreate(&ctx, 0, cu_dev);
        if (cuerr != CUDA_SUCCESS) {
            cu_log_error("cuCtxCreate", cuerr);
            return -1;
        }
    }

    int dev = 0;
    struct cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess) { cuda_log_error("cudaGetDevice", cudaGetLastError()); return -1; }
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) { cuda_log_error("cudaGetDeviceProperties", cudaGetLastError()); return -1; }
    char arch[64];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", prop.major, prop.minor);

    const char *src =
        "extern \"C\" __global__ void conv1d_kernel(const float* x, const float* w, const float* b, float* y, int in_ch, int out_ch, int T, int k, int stride, int groups, int out_len, int left_pad) {\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  int total = out_ch * out_len;\n"
        "  if (idx >= total) return;\n"
        "  int oc = idx / out_len;\n"
        "  int t = idx - oc * out_len;\n"
        "  int out_per_group = out_ch / groups;\n"
        "  int in_per_group = in_ch / groups;\n"
        "  int g = oc / out_per_group;\n"
        "  int in_base = g * in_per_group;\n"
        "  const float* wbase = w + ((long long)oc * in_per_group * k);\n"
        "  float sum = b ? b[oc] : 0.0f;\n"
        "  int in_start = t * stride - left_pad;\n"
        "  for (int ic = 0; ic < in_per_group; ic++) {\n"
        "    const float* wrow = wbase + ic * k;\n"
        "    const float* xch = x + ((long long)(in_base + ic) * T);\n"
        "    for (int kk = 0; kk < k; kk++) {\n"
        "      int xi = in_start + kk;\n"
        "      if (xi < 0 || xi >= T) continue;\n"
        "      sum += wrow[kk] * xch[xi];\n"
        "    }\n"
        "  }\n"
        "  y[((long long)oc * out_len + t)] = sum;\n"
        "}\n"
        "extern \"C\" __global__ void convtr1d_kernel(const float* x, const float* w, const float* b, float* y, int in_ch, int out_ch, int T, int k, int stride, int groups, int out_len) {\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  int total = out_ch * out_len;\n"
        "  if (idx >= total) return;\n"
        "  int oc = idx / out_len;\n"
        "  int t = idx - oc * out_len;\n"
        "  int out_per_group = out_ch / groups;\n"
        "  int in_per_group = in_ch / groups;\n"
        "  int g = oc / out_per_group;\n"
        "  int in_base = g * in_per_group;\n"
        "  float sum = b ? b[oc] : 0.0f;\n"
        "  for (int ic = 0; ic < in_per_group; ic++) {\n"
        "    const float* xch = x + ((long long)(in_base + ic) * T);\n"
        "    const float* wbase = w + ((long long)(in_base + ic) * out_per_group * k);\n"
        "    for (int kk = 0; kk < k; kk++) {\n"
        "      int ti = t - kk;\n"
        "      if (ti < 0) continue;\n"
        "      if (ti % stride) continue;\n"
        "      int xi = ti / stride;\n"
        "      if (xi < 0 || xi >= T) continue;\n"
        "      sum += wbase[(oc - g * out_per_group) * k + kk] * xch[xi];\n"
        "    }\n"
        "  }\n"
        "  y[((long long)oc * out_len + t)] = sum;\n"
        "}\n"
        "extern \"C\" __global__ void elu_kernel(float* x, int n) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  float v = x[i];\n"
        "  x[i] = v >= 0.0f ? v : expf(v) - 1.0f;\n"
        "}\n"
        "extern \"C\" __global__ void add_kernel(float* a, const float* b, int n) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  a[i] += b[i];\n"
        "}\n"
        "extern \"C\" __global__ void bias_kernel(float* x, const float* b, int n) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  x[i] += b[i];\n"
        "}\n"
        "extern \"C\" __global__ void silu_kernel(float* x, int n) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  float v = x[i];\n"
        "  x[i] = v / (1.0f + expf(-v));\n"
        "}\n"
        "extern \"C\" __global__ void add3_kernel(float* out, const float* a, const float* b, const float* c, int n, float scale) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  out[i] = (a[i] + b[i]) * scale + c[i];\n"
        "}\n"
        "extern \"C\" __global__ void affine_kernel(float* x, const float* scale, const float* shift, int n) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  x[i] = x[i] * (1.0f + scale[i]) + shift[i];\n"
        "}\n"
        "extern \"C\" __global__ void gate_add_kernel(float* x, const float* gate, const float* delta, int n) {\n"
        "  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (i >= n) return;\n"
        "  x[i] += gate[i] * delta[i];\n"
        "}\n"
        "extern \"C\" __global__ void layernorm_kernel(const float* x, const float* w, const float* b, float* y, int d, float eps) {\n"
        "  __shared__ float s_sum[256];\n"
        "  __shared__ float s_sumsq[256];\n"
        "  int tid = (int)threadIdx.x;\n"
        "  float sum = 0.0f;\n"
        "  float sumsq = 0.0f;\n"
        "  for (int i = tid; i < d; i += blockDim.x) {\n"
        "    float v = x[i];\n"
        "    sum += v;\n"
        "    sumsq += v * v;\n"
        "  }\n"
        "  s_sum[tid] = sum;\n"
        "  s_sumsq[tid] = sumsq;\n"
        "  __syncthreads();\n"
        "  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      s_sum[tid] += s_sum[tid + stride];\n"
        "      s_sumsq[tid] += s_sumsq[tid + stride];\n"
        "    }\n"
        "    __syncthreads();\n"
        "  }\n"
        "  float mean = s_sum[0] / (float)d;\n"
        "  float var = s_sumsq[0] / (float)d - mean * mean;\n"
        "  float inv = rsqrtf(var + eps);\n"
        "  for (int i = tid; i < d; i += blockDim.x) {\n"
        "    float v = (x[i] - mean) * inv;\n"
        "    if (w) v *= w[i];\n"
        "    if (b) v += b[i];\n"
        "    y[i] = v;\n"
        "  }\n"
        "}\n"
        "extern \"C\" __global__ void rmsnorm_kernel(const float* x, const float* alpha, float* y, int d, float eps) {\n"
        "  __shared__ float s_sum[256];\n"
        "  __shared__ float s_sumsq[256];\n"
        "  int tid = (int)threadIdx.x;\n"
        "  float sum = 0.0f;\n"
        "  float sumsq = 0.0f;\n"
        "  for (int i = tid; i < d; i += blockDim.x) {\n"
        "    float v = x[i];\n"
        "    sum += v;\n"
        "    sumsq += v * v;\n"
        "  }\n"
        "  s_sum[tid] = sum;\n"
        "  s_sumsq[tid] = sumsq;\n"
        "  __syncthreads();\n"
        "  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      s_sum[tid] += s_sum[tid + stride];\n"
        "      s_sumsq[tid] += s_sumsq[tid + stride];\n"
        "    }\n"
        "    __syncthreads();\n"
        "  }\n"
        "  float mean = s_sum[0] / (float)d;\n"
        "  float var = s_sumsq[0] - 2.0f * mean * s_sum[0] + mean * mean * (float)d;\n"
        "  if (d > 1) var /= (float)(d - 1);\n"
        "  else var = 0.0f;\n"
        "  float inv = rsqrtf(var + eps);\n"
        "  for (int i = tid; i < d; i += blockDim.x) {\n"
        "    float v = x[i] * (alpha ? alpha[i] : 1.0f) * inv;\n"
        "    y[i] = v;\n"
        "  }\n"
        "}\n"
        "extern \"C\" __global__ void attn_scores_kernel(const float* q, const float* k, float* scores, int T, int H, int D, float scale) {\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  int total = H * T * T;\n"
        "  if (idx >= total) return;\n"
        "  int tk = idx % T;\n"
        "  int tq = (idx / T) % T;\n"
        "  int h = idx / (T * T);\n"
        "  if (tk > tq) {\n"
        "    scores[idx] = -1e30f;\n"
        "    return;\n"
        "  }\n"
        "  const float* qv = q + ((tq * H + h) * D);\n"
        "  const float* kv = k + ((tk * H + h) * D);\n"
        "  float dot = 0.0f;\n"
        "  for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "  scores[idx] = dot * scale;\n"
        "}\n"
        "extern \"C\" __global__ void attn_softmax_kernel(float* scores, int T, int H) {\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  int total = H * T;\n"
        "  if (idx >= total) return;\n"
        "  int h = idx / T;\n"
        "  int tq = idx - h * T;\n"
        "  float maxv = -1e30f;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    float v = scores[(h * T + tq) * T + tk];\n"
        "    if (v > maxv) maxv = v;\n"
        "  }\n"
        "  float sum = 0.0f;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    float v = expf(scores[(h * T + tq) * T + tk] - maxv);\n"
        "    scores[(h * T + tq) * T + tk] = v;\n"
        "    sum += v;\n"
        "  }\n"
        "  float inv = 1.0f / sum;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    scores[(h * T + tq) * T + tk] *= inv;\n"
        "  }\n"
        "}\n"
        "extern \"C\" __global__ void attn_apply_kernel(const float* scores, const float* v, float* out, int T, int H, int D) {\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  int total = H * T;\n"
        "  if (idx >= total) return;\n"
        "  int h = idx / T;\n"
        "  int tq = idx - h * T;\n"
        "  float* outv = out + ((tq * H + h) * D);\n"
        "  for (int d = 0; d < D; d++) outv[d] = 0.0f;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    float w = scores[(h * T + tq) * T + tk];\n"
        "    const float* vv = v + ((tk * H + h) * D);\n"
        "    for (int d = 0; d < D; d++) outv[d] += w * vv[d];\n"
        "  }\n"
        "}\n"
        "extern \"C\" __global__ void attn_fused_kernel(const float* q, const float* k, const float* v, float* out, int T, int H, int D, float scale) {\n"
        "  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  int total = H * T;\n"
        "  if (idx >= total) return;\n"
        "  int h = idx / T;\n"
        "  int tq = idx - h * T;\n"
        "  const float* qv = q + ((tq * H + h) * D);\n"
        "  float maxv = -1e30f;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    const float* kv = k + ((tk * H + h) * D);\n"
        "    float dot = 0.0f;\n"
        "    for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "    float s = dot * scale;\n"
        "    if (s > maxv) maxv = s;\n"
        "  }\n"
        "  float sum = 0.0f;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    const float* kv = k + ((tk * H + h) * D);\n"
        "    float dot = 0.0f;\n"
        "    for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "    sum += expf(dot * scale - maxv);\n"
        "  }\n"
        "  float inv = 1.0f / sum;\n"
        "  float* outv = out + ((tq * H + h) * D);\n"
        "  for (int d = 0; d < D; d++) outv[d] = 0.0f;\n"
        "  for (int tk = 0; tk <= tq; tk++) {\n"
        "    const float* kv = k + ((tk * H + h) * D);\n"
        "    float dot = 0.0f;\n"
        "    for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "    float w = expf(dot * scale - maxv) * inv;\n"
        "    const float* vv = v + ((tk * H + h) * D);\n"
        "    for (int d = 0; d < D; d++) outv[d] += w * vv[d];\n"
        "  }\n"
        "}\n"
        "extern \"C\" __global__ void attn_step_kernel(const float* q, const float* k, const float* v, float* out, int T, int H, int D, float scale) {\n"
        "  int h = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
        "  if (h >= H) return;\n"
        "  const float* qv = q + h * D;\n"
        "  float maxv = -1e30f;\n"
        "  for (int tk = 0; tk < T; tk++) {\n"
        "    const float* kv = k + ((tk * H + h) * D);\n"
        "    float dot = 0.0f;\n"
        "    for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "    float s = dot * scale;\n"
        "    if (s > maxv) maxv = s;\n"
        "  }\n"
        "  float sum = 0.0f;\n"
        "  for (int tk = 0; tk < T; tk++) {\n"
        "    const float* kv = k + ((tk * H + h) * D);\n"
        "    float dot = 0.0f;\n"
        "    for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "    sum += expf(dot * scale - maxv);\n"
        "  }\n"
        "  float inv = 1.0f / sum;\n"
        "  float* outv = out + h * D;\n"
        "  for (int d = 0; d < D; d++) outv[d] = 0.0f;\n"
        "  for (int tk = 0; tk < T; tk++) {\n"
        "    const float* kv = k + ((tk * H + h) * D);\n"
        "    float dot = 0.0f;\n"
        "    for (int d = 0; d < D; d++) dot += qv[d] * kv[d];\n"
        "    float w = expf(dot * scale - maxv) * inv;\n"
        "    const float* vv = v + ((tk * H + h) * D);\n"
        "    for (int d = 0; d < D; d++) outv[d] += w * vv[d];\n"
        "  }\n"
        "}\n";

    nvrtcProgram prog;
    nvrtcResult rc = nvrtcCreateProgram(&prog, src, "ptts_kernels.cu", 0, NULL, NULL);
    if (rc != NVRTC_SUCCESS) return -1;
    const char *opts[] = {arch};
    rc = nvrtcCompileProgram(prog, 1, opts);
    if (rc != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            if (log) {
                nvrtcGetProgramLog(prog, log);
                fprintf(stderr, "[ptts] NVRTC compile log:\n%s\n", log);
                free(log);
            }
        }
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

    cuerr = cuModuleLoadData(&g_mod, ptx);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuModuleLoadData", cuerr);
        free(ptx);
        return -1;
    }
    free(ptx);
    cuerr = cuModuleGetFunction(&g_conv1d, g_mod, "conv1d_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(conv1d)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_convtr1d, g_mod, "convtr1d_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(convtr1d)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_elu, g_mod, "elu_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(elu)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_add, g_mod, "add_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(add)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_bias, g_mod, "bias_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(bias)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_silu, g_mod, "silu_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(silu)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_add3, g_mod, "add3_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(add3)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_affine, g_mod, "affine_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(affine)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_gate_add, g_mod, "gate_add_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(gate_add)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_layernorm, g_mod, "layernorm_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(layernorm)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_rmsnorm, g_mod, "rmsnorm_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(rmsnorm)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_attn_scores, g_mod, "attn_scores_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(attn_scores)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_attn_softmax, g_mod, "attn_softmax_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(attn_softmax)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_attn_apply, g_mod, "attn_apply_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(attn_apply)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_attn_fused, g_mod, "attn_fused_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(attn_fused)", cuerr); return -1; }
    cuerr = cuModuleGetFunction(&g_attn_step, g_mod, "attn_step_kernel");
    if (cuerr != CUDA_SUCCESS) { cu_log_error("cuModuleGetFunction(attn_step)", cuerr); return -1; }

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
    if (cudaMalloc((void **)buf, bytes) != cudaSuccess) {
        cuda_log_error("cudaMalloc", cudaGetLastError());
        return -1;
    }
    *cap = bytes;
    return 0;
}

static int ensure_conv_buffer(float **buf, size_t *cap, size_t bytes) {
    return ensure_device_buffer(buf, cap, bytes);
}

static int ensure_kv_cache(int max_len) {
    if (g_kv_cache_max_len >= max_len && g_kv_cache_bytes > 0) return 0;
    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        if (g_k_cache_dev[i]) cudaFree(g_k_cache_dev[i]);
        if (g_v_cache_dev[i]) cudaFree(g_v_cache_dev[i]);
        g_k_cache_dev[i] = NULL;
        g_v_cache_dev[i] = NULL;
    }
    size_t elems = (size_t)max_len * FLOWLM_NUM_HEADS * FLOWLM_HEAD_DIM;
    size_t bytes = elems * sizeof(float);
    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        if (cudaMalloc((void **)&g_k_cache_dev[i], bytes) != cudaSuccess) {
            cuda_log_error("cudaMalloc(k_cache)", cudaGetLastError());
            return -1;
        }
        if (cudaMalloc((void **)&g_v_cache_dev[i], bytes) != cudaSuccess) {
            cuda_log_error("cudaMalloc(v_cache)", cudaGetLastError());
            return -1;
        }
    }
    g_kv_cache_bytes = bytes;
    g_kv_cache_max_len = max_len;
    return 0;
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
    if (cudaMalloc((void **)&dev, bytes) != cudaSuccess) { cuda_log_error("cudaMalloc(weight)", cudaGetLastError()); return NULL; }
    if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(weight)", cudaGetLastError());
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
    if (cudaMemcpy(g_tmp_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(x)", cudaGetLastError());
        return -1;
    }

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
    if (cuda_check("cublasSgemm") != 0) return -1;

    if (cudaMemcpy(y, g_tmp_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(y)", cudaGetLastError());
        return -1;
    }

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
    if (cudaMemcpy(g_tmp_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(conv1d x)", cudaGetLastError());
        return -1;
    }

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
    CUresult cuerr = cuLaunchKernel(g_conv1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(conv1d)", cuerr);
        return -1;
    }
    if (cudaMemcpy(y, g_tmp_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(conv1d y)", cudaGetLastError());
        return -1;
    }
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
    if (cudaMemcpy(g_tmp_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convtr1d x)", cudaGetLastError());
        return -1;
    }

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
    CUresult cuerr = cuLaunchKernel(g_convtr1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(convtr1d)", cuerr);
        return -1;
    }
    if (cudaMemcpy(y, g_tmp_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convtr1d y)", cudaGetLastError());
        return -1;
    }
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
    CUresult cuerr = cuLaunchKernel(g_conv1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(conv1d_device)", cuerr);
        return -1;
    }
    if (cuda_check("conv1d_device") != 0) return -1;
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
    CUresult cuerr = cuLaunchKernel(g_convtr1d, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(convtr1d_device)", cuerr);
        return -1;
    }
    if (cuda_check("convtr1d_device") != 0) return -1;
    return 0;
}

static int elu_device(float *x, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&x, &n};
    CUresult cuerr = cuLaunchKernel(g_elu, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(elu_device)", cuerr);
        return -1;
    }
    if (cuda_check("elu_device") != 0) return -1;
    return 0;
}

static int add_device(float *a, const float *b, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&a, &b, &n};
    CUresult cuerr = cuLaunchKernel(g_add, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(add_device)", cuerr);
        return -1;
    }
    if (cuda_check("add_device") != 0) return -1;
    return 0;
}

static int bias_device(float *x, const float *b, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&x, &b, &n};
    CUresult cuerr = cuLaunchKernel(g_bias, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(bias)", cuerr);
        return -1;
    }
    if (cuda_check("bias_device") != 0) return -1;
    return 0;
}

static int silu_device(float *x, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&x, &n};
    CUresult cuerr = cuLaunchKernel(g_silu, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(silu)", cuerr);
        return -1;
    }
    if (cuda_check("silu_device") != 0) return -1;
    return 0;
}

static int add3_device(float *out, const float *a, const float *b, const float *c, int n, float scale) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&out, &a, &b, &c, &n, &scale};
    CUresult cuerr = cuLaunchKernel(g_add3, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(add3)", cuerr);
        return -1;
    }
    if (cuda_check("add3_device") != 0) return -1;
    return 0;
}

static int affine_device(float *x, const float *scale, const float *shift, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&x, &scale, &shift, &n};
    CUresult cuerr = cuLaunchKernel(g_affine, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(affine)", cuerr);
        return -1;
    }
    if (cuda_check("affine_device") != 0) return -1;
    return 0;
}

static int gate_add_device(float *x, const float *gate, const float *delta, int n) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    void *args[] = {&x, &gate, &delta, &n};
    CUresult cuerr = cuLaunchKernel(g_gate_add, grid, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(gate_add)", cuerr);
        return -1;
    }
    if (cuda_check("gate_add_device") != 0) return -1;
    return 0;
}

static int layernorm_device(const float *x, const float *w, const float *b, float *y, int d, float eps) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    void *args[] = {&x, &w, &b, &y, &d, &eps};
    CUresult cuerr = cuLaunchKernel(g_layernorm, 1, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(layernorm)", cuerr);
        return -1;
    }
    if (cuda_check("layernorm_device") != 0) return -1;
    return 0;
}

static int rmsnorm_device(const float *x, const float *alpha, float *y, int d, float eps) {
    if (ensure_kernels() != 0) return -1;
    int block = 256;
    void *args[] = {&x, &alpha, &y, &d, &eps};
    CUresult cuerr = cuLaunchKernel(g_rmsnorm, 1, 1, 1, block, 1, 1, 0, 0, args, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(rmsnorm)", cuerr);
        return -1;
    }
    if (cuda_check("rmsnorm_device") != 0) return -1;
    return 0;
}

static int linear_device(float *d_y, const float *d_x, const float *w, const float *b,
                         int in, int out) {
    if (ensure_init() != 0) return -1;
    float *d_w = get_weight_device(w, (size_t)out * (size_t)in * sizeof(float));
    if (!d_w) return -1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out, 1, in,
        &alpha,
        d_w, in,
        d_x, in,
        &beta,
        d_y, out);
    if (st != CUBLAS_STATUS_SUCCESS) return -1;
    if (cuda_check("cublasSgemm_device") != 0) return -1;
    if (b) {
        float *d_b = get_weight_device(b, (size_t)out * sizeof(float));
        if (!d_b) return -1;
        if (bias_device(d_y, d_b, out) != 0) return -1;
    }
    return 0;
}

static int convtr1d_host_fallback(float *d_y, const float *d_x,
                                  const float *w, const float *b,
                                  int in_ch, int out_ch, int T, int k, int stride, int groups,
                                  int out_len) {
    size_t x_bytes = (size_t)in_ch * T * sizeof(float);
    size_t y_bytes = (size_t)out_ch * out_len * sizeof(float);
    float *h_x = (float *)malloc(x_bytes);
    float *h_y = (float *)malloc(y_bytes);
    if (!h_x || !h_y) { free(h_x); free(h_y); return -1; }
    if (cudaMemcpy(h_x, d_x, x_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convtr host x)", cudaGetLastError());
        free(h_x); free(h_y); return -1;
    }
    ptts_convtr1d_forward(h_y, h_x, w, b, in_ch, out_ch, T, k, stride, groups);
    if (cudaMemcpy(d_y, h_y, y_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convtr host y)", cudaGetLastError());
        free(h_x); free(h_y); return -1;
    }
    free(h_x);
    free(h_y);
    return 0;
}

static int conv1d_host_fallback(float *d_y, const float *d_x,
                                const float *w, const float *b,
                                int in_ch, int out_ch, int T, int k, int stride, int groups,
                                int out_len) {
    size_t x_bytes = (size_t)in_ch * T * sizeof(float);
    size_t y_bytes = (size_t)out_ch * out_len * sizeof(float);
    float *h_x = (float *)malloc(x_bytes);
    float *h_y = (float *)malloc(y_bytes);
    if (!h_x || !h_y) { free(h_x); free(h_y); return -1; }
    if (cudaMemcpy(h_x, d_x, x_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(conv1d host x)", cudaGetLastError());
        free(h_x); free(h_y); return -1;
    }
    ptts_conv1d_forward(h_y, h_x, w, b, in_ch, out_ch, T, k, stride, groups);
    if (cudaMemcpy(d_y, h_y, y_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(conv1d host y)", cudaGetLastError());
        free(h_x); free(h_y); return -1;
    }
    free(h_x);
    free(h_y);
    return 0;
}

static void cpu_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                               int in_ch, int out_ch, int T, int k, int stride, int groups) {
    int out_len = T / stride;
    int in_per_group = in_ch / groups;
    int out_per_group = out_ch / groups;
    int left_pad = k - stride;

    for (int oc = 0; oc < out_ch; oc++) {
        int g = oc / out_per_group;
        int in_base = g * in_per_group;
        const float *wbase = w + (size_t)oc * in_per_group * k;
        float bias = b ? b[oc] : 0.0f;
        for (int t = 0; t < out_len; t++) {
            float sum = bias;
            int in_start = t * stride - left_pad;
            for (int ic = 0; ic < in_per_group; ic++) {
                const float *wrow = wbase + ic * k;
                const float *xch = x + (size_t)(in_base + ic) * T;
                for (int kk = 0; kk < k; kk++) {
                    int idx = in_start + kk;
                    if (idx < 0 || idx >= T) continue;
                    sum += wrow[kk] * xch[idx];
                }
            }
            y[(size_t)oc * out_len + t] = sum;
        }
    }
}

static void cpu_convtr1d_forward(float *y, const float *x, const float *w, const float *b,
                                 int in_ch, int out_ch, int T, int k, int stride, int groups) {
    int full_len = (T - 1) * stride + k;
    int out_len = full_len - (k - stride);
    int out_per_group = out_ch / groups;
    int in_per_group = in_ch / groups;

    memset(y, 0, (size_t)out_ch * out_len * sizeof(float));

    for (int ic = 0; ic < in_ch; ic++) {
        int g = ic / in_per_group;
        int out_base = g * out_per_group;
        const float *wbase = w + (size_t)ic * out_per_group * k;
        const float *xch = x + (size_t)ic * T;
        for (int t = 0; t < T; t++) {
            int out_start = t * stride;
            for (int ocg = 0; ocg < out_per_group; ocg++) {
                const float *wrow = wbase + ocg * k;
                float *ych = y + (size_t)(out_base + ocg) * out_len;
                for (int kk = 0; kk < k; kk++) {
                    int idx = out_start + kk;
                    if (idx >= out_len) continue;
                    ych[idx] += wrow[kk] * xch[t];
                }
            }
        }
    }
    if (b) {
        for (int oc = 0; oc < out_ch; oc++) {
            float bias = b[oc];
            float *ych = y + (size_t)oc * out_len;
            for (int t = 0; t < out_len; t++) ych[t] += bias;
        }
    }
}

static void cpu_elu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v >= 0.0f ? v : expf(v) - 1.0f;
    }
}

static void cpu_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

static int ensure_host_buffer(float **buf, size_t *cap, size_t bytes) {
    if (*cap >= bytes && *buf) return 0;
    float *next = (float *)realloc(*buf, bytes);
    if (!next) return -1;
    *buf = next;
    *cap = bytes;
    return 0;
}

static void compare_gpu_cpu(const char *label, const float *d_buf, const float *h_ref,
                            int n, float **h_tmp, size_t *cap) {
    size_t bytes = (size_t)n * sizeof(float);
    if (ensure_host_buffer(h_tmp, cap, bytes) != 0) return;
    if (cudaMemcpy(*h_tmp, d_buf, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(compare)", cudaGetLastError());
        return;
    }
    float maxd = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = (*h_tmp)[i] - h_ref[i];
        if (d < 0.0f) d = -d;
        if (d > maxd) maxd = d;
    }
    fprintf(stderr, "[ptts] CUDA validate %-10s maxdiff=%.6f\n", label, maxd);
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
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, x_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, x_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf3, &g_conv_buf3_bytes, x_bytes) != 0) return -1;
    float *d_x = g_conv_buf0;
    float *d_y = g_conv_buf1;
    float *d_tmp1 = g_conv_buf2;
    float *d_tmp2 = g_conv_buf3;
    float *h_x = NULL;
    float *h_y = NULL;
    float *h_tmp1 = NULL;
    float *h_tmp2 = NULL;
    float *h_gpu = NULL;
    size_t h_x_cap = 0;
    size_t h_y_cap = 0;
    size_t h_tmp1_cap = 0;
    size_t h_tmp2_cap = 0;
    size_t h_gpu_cap = 0;
    int validate = cuda_validate_enabled();
    if (validate) {
        if (ensure_host_buffer(&h_x, &h_x_cap, x_bytes) != 0) return -1;
        memcpy(h_x, x_host, x_bytes);
    }
    if (cudaMemcpy(d_x, x_host, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convstack x)", cudaGetLastError());
        return -1;
    }

    int t = T;
    int out_t = t / dec_in->stride;
    size_t y_bytes = (size_t)dec_in->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf1;
    float *d_w = get_weight_device(dec_in->w, (size_t)dec_in->out_ch * (dec_in->in_ch / dec_in->groups) * dec_in->k * sizeof(float));
    float *d_b = dec_in->b ? get_weight_device(dec_in->b, (size_t)dec_in->out_ch * sizeof(float)) : NULL;
    if (!d_w) return -1;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: dec_in\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_y, d_x, d_w, d_b, dec_in->in_ch, dec_in->out_ch, t, dec_in->k, dec_in->stride, dec_in->groups) != 0) {
            return -1;
        }
    } else {
        if (conv1d_host_fallback(d_y, d_x, dec_in->w, dec_in->b, dec_in->in_ch, dec_in->out_ch,
                                 t, dec_in->k, dec_in->stride, dec_in->groups, out_t) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)dec_in->out_ch * out_t * sizeof(float);
        if (ensure_host_buffer(&h_y, &h_y_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_y, h_x, dec_in->w, dec_in->b, dec_in->in_ch, dec_in->out_ch, t,
                           dec_in->k, dec_in->stride, dec_in->groups);
        compare_gpu_cpu("dec_in", d_y, h_y, dec_in->out_ch * out_t, &h_gpu, &h_gpu_cap);
        float *swap = h_x; h_x = h_y; h_y = swap;
    }
    d_x = d_y;
    t = out_t;

    if (elu_device(d_x, dec_in->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_x, dec_in->out_ch * t);
        compare_gpu_cpu("elu0", d_x, h_x, dec_in->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    out_t = t * up0->stride;
    y_bytes = (size_t)up0->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf0;
    d_w = get_weight_device(up0->w, (size_t)up0->in_ch * (up0->out_ch / up0->groups) * up0->k * sizeof(float));
    d_b = up0->b ? get_weight_device(up0->b, (size_t)up0->out_ch * sizeof(float)) : NULL;
    if (!d_w) return -1;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: up0\n");
    if (cuda_convtr_enabled()) {
        if (convtr1d_device(d_y, d_x, d_w, d_b, up0->in_ch, up0->out_ch, t, up0->k, up0->stride, up0->groups) != 0) {
            return -1;
        }
    } else {
        if (convtr1d_host_fallback(d_y, d_x, up0->w, up0->b, up0->in_ch, up0->out_ch,
                                   t, up0->k, up0->stride, up0->groups, out_t) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)up0->out_ch * out_t * sizeof(float);
        if (ensure_host_buffer(&h_y, &h_y_cap, bytes) != 0) return -1;
        cpu_convtr1d_forward(h_y, h_x, up0->w, up0->b, up0->in_ch, up0->out_ch, t,
                             up0->k, up0->stride, up0->groups);
        compare_gpu_cpu("up0", d_y, h_y, up0->out_ch * out_t, &h_gpu, &h_gpu_cap);
        float *swap = h_x; h_x = h_y; h_y = swap;
    }
    d_x = d_y;
    t = out_t;

    size_t tmp1_bytes = (size_t)res0_1->in_ch * t * sizeof(float);
    size_t tmp2_bytes = (size_t)res0_1->out_ch * t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, tmp1_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf3, &g_conv_buf3_bytes, tmp2_bytes) != 0) return -1;
    d_tmp1 = g_conv_buf2;
    d_tmp2 = g_conv_buf3;
    if (cudaMemcpy(d_tmp1, d_x, tmp1_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convstack tmp1)", cudaGetLastError());
        return -1;
    }
    if (elu_device(d_tmp1, res0_1->in_ch * t) != 0) return -1;
    if (validate) {
        if (ensure_host_buffer(&h_tmp1, &h_tmp1_cap, tmp1_bytes) != 0) return -1;
        memcpy(h_tmp1, h_x, tmp1_bytes);
        cpu_elu_inplace(h_tmp1, res0_1->in_ch * t);
        compare_gpu_cpu("res0_elu1", d_tmp1, h_tmp1, res0_1->in_ch * t, &h_gpu, &h_gpu_cap);
    }
    d_w = get_weight_device(res0_1->w, (size_t)res0_1->out_ch * (res0_1->in_ch / res0_1->groups) * res0_1->k * sizeof(float));
    d_b = res0_1->b ? get_weight_device(res0_1->b, (size_t)res0_1->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: res0_1\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_tmp2, d_tmp1, d_w, d_b, res0_1->in_ch, res0_1->out_ch, t, res0_1->k, res0_1->stride, res0_1->groups) != 0) {
            return -1;
        }
    } else {
        int out_len = t / res0_1->stride;
        if (conv1d_host_fallback(d_tmp2, d_tmp1, res0_1->w, res0_1->b, res0_1->in_ch, res0_1->out_ch,
                                 t, res0_1->k, res0_1->stride, res0_1->groups, out_len) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)res0_1->out_ch * t * sizeof(float);
        if (ensure_host_buffer(&h_tmp2, &h_tmp2_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_tmp2, h_tmp1, res0_1->w, res0_1->b, res0_1->in_ch, res0_1->out_ch,
                           t, res0_1->k, res0_1->stride, res0_1->groups);
        compare_gpu_cpu("res0_1", d_tmp2, h_tmp2, res0_1->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    if (elu_device(d_tmp2, res0_1->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_tmp2, res0_1->out_ch * t);
        compare_gpu_cpu("res0_elu2", d_tmp2, h_tmp2, res0_1->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    d_w = get_weight_device(res0_2->w, (size_t)res0_2->out_ch * (res0_2->in_ch / res0_2->groups) * res0_2->k * sizeof(float));
    d_b = res0_2->b ? get_weight_device(res0_2->b, (size_t)res0_2->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: res0_2\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_tmp1, d_tmp2, d_w, d_b, res0_2->in_ch, res0_2->out_ch, t, res0_2->k, res0_2->stride, res0_2->groups) != 0) {
            return -1;
        }
    } else {
        int out_len = t / res0_2->stride;
        if (conv1d_host_fallback(d_tmp1, d_tmp2, res0_2->w, res0_2->b, res0_2->in_ch, res0_2->out_ch,
                                 t, res0_2->k, res0_2->stride, res0_2->groups, out_len) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)res0_2->out_ch * t * sizeof(float);
        if (ensure_host_buffer(&h_tmp1, &h_tmp1_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_tmp1, h_tmp2, res0_2->w, res0_2->b, res0_2->in_ch, res0_2->out_ch,
                           t, res0_2->k, res0_2->stride, res0_2->groups);
        compare_gpu_cpu("res0_2", d_tmp1, h_tmp1, res0_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    if (add_device(d_x, d_tmp1, res0_2->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_add_inplace(h_x, h_tmp1, res0_2->out_ch * t);
        compare_gpu_cpu("res0_add", d_x, h_x, res0_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    if (elu_device(d_x, res0_2->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_x, res0_2->out_ch * t);
        compare_gpu_cpu("res0_elu3", d_x, h_x, res0_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    out_t = t * up1->stride;
    y_bytes = (size_t)up1->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf1;
    d_w = get_weight_device(up1->w, (size_t)up1->in_ch * (up1->out_ch / up1->groups) * up1->k * sizeof(float));
    d_b = up1->b ? get_weight_device(up1->b, (size_t)up1->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: up1\n");
    if (cuda_convtr_enabled()) {
        if (convtr1d_device(d_y, d_x, d_w, d_b, up1->in_ch, up1->out_ch, t, up1->k, up1->stride, up1->groups) != 0) {
            return -1;
        }
    } else {
        if (convtr1d_host_fallback(d_y, d_x, up1->w, up1->b, up1->in_ch, up1->out_ch,
                                   t, up1->k, up1->stride, up1->groups, out_t) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)up1->out_ch * out_t * sizeof(float);
        if (ensure_host_buffer(&h_y, &h_y_cap, bytes) != 0) return -1;
        cpu_convtr1d_forward(h_y, h_x, up1->w, up1->b, up1->in_ch, up1->out_ch, t,
                             up1->k, up1->stride, up1->groups);
        compare_gpu_cpu("up1", d_y, h_y, up1->out_ch * out_t, &h_gpu, &h_gpu_cap);
        float *swap = h_x; h_x = h_y; h_y = swap;
    }
    d_x = d_y;
    t = out_t;

    tmp1_bytes = (size_t)res1_1->in_ch * t * sizeof(float);
    tmp2_bytes = (size_t)res1_1->out_ch * t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, tmp1_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf3, &g_conv_buf3_bytes, tmp2_bytes) != 0) return -1;
    d_tmp1 = g_conv_buf2;
    d_tmp2 = g_conv_buf3;
    if (cudaMemcpy(d_tmp1, d_x, tmp1_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convstack tmp1 stage1)", cudaGetLastError());
        return -1;
    }
    if (elu_device(d_tmp1, res1_1->in_ch * t) != 0) return -1;
    if (validate) {
        if (ensure_host_buffer(&h_tmp1, &h_tmp1_cap, tmp1_bytes) != 0) return -1;
        memcpy(h_tmp1, h_x, tmp1_bytes);
        cpu_elu_inplace(h_tmp1, res1_1->in_ch * t);
        compare_gpu_cpu("res1_elu1", d_tmp1, h_tmp1, res1_1->in_ch * t, &h_gpu, &h_gpu_cap);
    }
    d_w = get_weight_device(res1_1->w, (size_t)res1_1->out_ch * (res1_1->in_ch / res1_1->groups) * res1_1->k * sizeof(float));
    d_b = res1_1->b ? get_weight_device(res1_1->b, (size_t)res1_1->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: res1_1\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_tmp2, d_tmp1, d_w, d_b, res1_1->in_ch, res1_1->out_ch, t, res1_1->k, res1_1->stride, res1_1->groups) != 0) {
            return -1;
        }
    } else {
        int out_len = t / res1_1->stride;
        if (conv1d_host_fallback(d_tmp2, d_tmp1, res1_1->w, res1_1->b, res1_1->in_ch, res1_1->out_ch,
                                 t, res1_1->k, res1_1->stride, res1_1->groups, out_len) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)res1_1->out_ch * t * sizeof(float);
        if (ensure_host_buffer(&h_tmp2, &h_tmp2_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_tmp2, h_tmp1, res1_1->w, res1_1->b, res1_1->in_ch, res1_1->out_ch,
                           t, res1_1->k, res1_1->stride, res1_1->groups);
        compare_gpu_cpu("res1_1", d_tmp2, h_tmp2, res1_1->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    if (elu_device(d_tmp2, res1_1->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_tmp2, res1_1->out_ch * t);
        compare_gpu_cpu("res1_elu2", d_tmp2, h_tmp2, res1_1->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    d_w = get_weight_device(res1_2->w, (size_t)res1_2->out_ch * (res1_2->in_ch / res1_2->groups) * res1_2->k * sizeof(float));
    d_b = res1_2->b ? get_weight_device(res1_2->b, (size_t)res1_2->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: res1_2\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_tmp1, d_tmp2, d_w, d_b, res1_2->in_ch, res1_2->out_ch, t, res1_2->k, res1_2->stride, res1_2->groups) != 0) {
            return -1;
        }
    } else {
        int out_len = t / res1_2->stride;
        if (conv1d_host_fallback(d_tmp1, d_tmp2, res1_2->w, res1_2->b, res1_2->in_ch, res1_2->out_ch,
                                 t, res1_2->k, res1_2->stride, res1_2->groups, out_len) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)res1_2->out_ch * t * sizeof(float);
        if (ensure_host_buffer(&h_tmp1, &h_tmp1_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_tmp1, h_tmp2, res1_2->w, res1_2->b, res1_2->in_ch, res1_2->out_ch,
                           t, res1_2->k, res1_2->stride, res1_2->groups);
        compare_gpu_cpu("res1_2", d_tmp1, h_tmp1, res1_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    if (add_device(d_x, d_tmp1, res1_2->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_add_inplace(h_x, h_tmp1, res1_2->out_ch * t);
        compare_gpu_cpu("res1_add", d_x, h_x, res1_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    if (elu_device(d_x, res1_2->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_x, res1_2->out_ch * t);
        compare_gpu_cpu("res1_elu3", d_x, h_x, res1_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    out_t = t * up2->stride;
    y_bytes = (size_t)up2->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf0, &g_conv_buf0_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf0;
    d_w = get_weight_device(up2->w, (size_t)up2->in_ch * (up2->out_ch / up2->groups) * up2->k * sizeof(float));
    d_b = up2->b ? get_weight_device(up2->b, (size_t)up2->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: up2\n");
    if (cuda_convtr_enabled()) {
        if (convtr1d_device(d_y, d_x, d_w, d_b, up2->in_ch, up2->out_ch, t, up2->k, up2->stride, up2->groups) != 0) {
            return -1;
        }
    } else {
        if (convtr1d_host_fallback(d_y, d_x, up2->w, up2->b, up2->in_ch, up2->out_ch,
                                   t, up2->k, up2->stride, up2->groups, out_t) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)up2->out_ch * out_t * sizeof(float);
        if (ensure_host_buffer(&h_y, &h_y_cap, bytes) != 0) return -1;
        cpu_convtr1d_forward(h_y, h_x, up2->w, up2->b, up2->in_ch, up2->out_ch, t,
                             up2->k, up2->stride, up2->groups);
        compare_gpu_cpu("up2", d_y, h_y, up2->out_ch * out_t, &h_gpu, &h_gpu_cap);
        float *swap = h_x; h_x = h_y; h_y = swap;
    }
    d_x = d_y;
    t = out_t;

    tmp1_bytes = (size_t)res2_1->in_ch * t * sizeof(float);
    tmp2_bytes = (size_t)res2_1->out_ch * t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf2, &g_conv_buf2_bytes, tmp1_bytes) != 0) return -1;
    if (ensure_conv_buffer(&g_conv_buf3, &g_conv_buf3_bytes, tmp2_bytes) != 0) return -1;
    d_tmp1 = g_conv_buf2;
    d_tmp2 = g_conv_buf3;
    if (cudaMemcpy(d_tmp1, d_x, tmp1_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convstack tmp1 stage2)", cudaGetLastError());
        return -1;
    }
    if (elu_device(d_tmp1, res2_1->in_ch * t) != 0) return -1;
    if (validate) {
        if (ensure_host_buffer(&h_tmp1, &h_tmp1_cap, tmp1_bytes) != 0) return -1;
        memcpy(h_tmp1, h_x, tmp1_bytes);
        cpu_elu_inplace(h_tmp1, res2_1->in_ch * t);
        compare_gpu_cpu("res2_elu1", d_tmp1, h_tmp1, res2_1->in_ch * t, &h_gpu, &h_gpu_cap);
    }
    d_w = get_weight_device(res2_1->w, (size_t)res2_1->out_ch * (res2_1->in_ch / res2_1->groups) * res2_1->k * sizeof(float));
    d_b = res2_1->b ? get_weight_device(res2_1->b, (size_t)res2_1->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: res2_1\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_tmp2, d_tmp1, d_w, d_b, res2_1->in_ch, res2_1->out_ch, t, res2_1->k, res2_1->stride, res2_1->groups) != 0) {
            return -1;
        }
    } else {
        int out_len = t / res2_1->stride;
        if (conv1d_host_fallback(d_tmp2, d_tmp1, res2_1->w, res2_1->b, res2_1->in_ch, res2_1->out_ch,
                                 t, res2_1->k, res2_1->stride, res2_1->groups, out_len) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)res2_1->out_ch * t * sizeof(float);
        if (ensure_host_buffer(&h_tmp2, &h_tmp2_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_tmp2, h_tmp1, res2_1->w, res2_1->b, res2_1->in_ch, res2_1->out_ch,
                           t, res2_1->k, res2_1->stride, res2_1->groups);
        compare_gpu_cpu("res2_1", d_tmp2, h_tmp2, res2_1->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    if (elu_device(d_tmp2, res2_1->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_tmp2, res2_1->out_ch * t);
        compare_gpu_cpu("res2_elu2", d_tmp2, h_tmp2, res2_1->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    d_w = get_weight_device(res2_2->w, (size_t)res2_2->out_ch * (res2_2->in_ch / res2_2->groups) * res2_2->k * sizeof(float));
    d_b = res2_2->b ? get_weight_device(res2_2->b, (size_t)res2_2->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: res2_2\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_tmp1, d_tmp2, d_w, d_b, res2_2->in_ch, res2_2->out_ch, t, res2_2->k, res2_2->stride, res2_2->groups) != 0) {
            return -1;
        }
    } else {
        int out_len = t / res2_2->stride;
        if (conv1d_host_fallback(d_tmp1, d_tmp2, res2_2->w, res2_2->b, res2_2->in_ch, res2_2->out_ch,
                                 t, res2_2->k, res2_2->stride, res2_2->groups, out_len) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)res2_2->out_ch * t * sizeof(float);
        if (ensure_host_buffer(&h_tmp1, &h_tmp1_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_tmp1, h_tmp2, res2_2->w, res2_2->b, res2_2->in_ch, res2_2->out_ch,
                           t, res2_2->k, res2_2->stride, res2_2->groups);
        compare_gpu_cpu("res2_2", d_tmp1, h_tmp1, res2_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }
    if (add_device(d_x, d_tmp1, res2_2->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_add_inplace(h_x, h_tmp1, res2_2->out_ch * t);
        compare_gpu_cpu("res2_add", d_x, h_x, res2_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    if (elu_device(d_x, res2_2->out_ch * t) != 0) return -1;
    if (validate) {
        cpu_elu_inplace(h_x, res2_2->out_ch * t);
        compare_gpu_cpu("res2_elu3", d_x, h_x, res2_2->out_ch * t, &h_gpu, &h_gpu_cap);
    }

    out_t = t / dec_out->stride;
    y_bytes = (size_t)dec_out->out_ch * out_t * sizeof(float);
    if (ensure_conv_buffer(&g_conv_buf1, &g_conv_buf1_bytes, y_bytes) != 0) return -1;
    d_y = g_conv_buf1;
    d_w = get_weight_device(dec_out->w, (size_t)dec_out->out_ch * (dec_out->in_ch / dec_out->groups) * dec_out->k * sizeof(float));
    d_b = dec_out->b ? get_weight_device(dec_out->b, (size_t)dec_out->out_ch * sizeof(float)) : NULL;
    if (cuda_debug_enabled()) fprintf(stderr, "[ptts] CUDA convstack: dec_out\n");
    if (cuda_conv1d_enabled()) {
        if (conv1d_device(d_y, d_x, d_w, d_b, dec_out->in_ch, dec_out->out_ch, t, dec_out->k, dec_out->stride, dec_out->groups) != 0) {
            return -1;
        }
    } else {
        if (conv1d_host_fallback(d_y, d_x, dec_out->w, dec_out->b, dec_out->in_ch, dec_out->out_ch,
                                 t, dec_out->k, dec_out->stride, dec_out->groups, out_t) != 0) {
            return -1;
        }
    }
    if (validate) {
        size_t bytes = (size_t)dec_out->out_ch * out_t * sizeof(float);
        if (ensure_host_buffer(&h_y, &h_y_cap, bytes) != 0) return -1;
        cpu_conv1d_forward(h_y, h_x, dec_out->w, dec_out->b, dec_out->in_ch, dec_out->out_ch,
                           t, dec_out->k, dec_out->stride, dec_out->groups);
        compare_gpu_cpu("dec_out", d_y, h_y, dec_out->out_ch * out_t, &h_gpu, &h_gpu_cap);
        float *swap = h_x; h_x = h_y; h_y = swap;
    }

    if (cudaMemcpy(out_host, d_y, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_log_error("cudaMemcpy(convstack out)", cudaGetLastError());
        return -1;
    }
    if (validate) {
        free(h_x);
        free(h_y);
        free(h_tmp1);
        free(h_tmp2);
        free(h_gpu);
    }
    *out_len = out_t;
    return 0;
}

int ptts_cuda_flownet_forward(const ptts_cuda_flow_net_desc *desc,
                              const float *cond, const float *ts, const float *tt,
                              const float *x_in, float *out) {
    if (!desc || !cond || !ts || !tt || !x_in || !out) return -1;
    if (ensure_kernels() != 0) return -1;
    if (ensure_init() != 0) return -1;

    if (ensure_device_buffer(&g_flow_buf0, &g_flow_buf0_bytes, FLOWLM_LATENT_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf1, &g_flow_buf1_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf2, &g_flow_buf2_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf3, &g_flow_buf3_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf4, &g_flow_buf4_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf5, &g_flow_buf5_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf6, &g_flow_buf6_bytes, FLOWLM_FLOW_DIM * 3 * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf7, &g_flow_buf7_bytes, FLOWLM_FLOW_DIM * 2 * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf8, &g_flow_buf8_bytes, FLOWLM_D_MODEL * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf9, &g_flow_buf9_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;
    if (ensure_device_buffer(&g_flow_buf10, &g_flow_buf10_bytes, FLOWLM_FLOW_DIM * sizeof(float)) != 0) return -1;

    float *d_xin = g_flow_buf0;
    float *d_x = g_flow_buf1;
    float *d_tmp = g_flow_buf2;
    float *d_tmp2 = g_flow_buf3;
    float *d_tmp3 = g_flow_buf4;
    float *d_mlp = g_flow_buf5;
    float *d_ada = g_flow_buf6;
    float *d_ada2 = g_flow_buf7;
    float *d_cond = g_flow_buf8;
    float *d_ts = g_flow_buf9;
    float *d_tt = g_flow_buf10;

    int profile = flow_profile_enabled();
    cudaEvent_t ev_start, ev_stop;
    if (profile) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
    }

    if (cudaMemcpy(d_xin, x_in, FLOWLM_LATENT_DIM * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(d_cond, cond, FLOWLM_D_MODEL * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(d_ts, ts, FLOWLM_FLOW_DIM * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(d_tt, tt, FLOWLM_FLOW_DIM * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    if (profile) cudaEventRecord(ev_start, 0);
    if (linear_device(d_x, d_xin, desc->input_w, desc->input_b, FLOWLM_LATENT_DIM, FLOWLM_FLOW_DIM) != 0) return -1;
    if (profile) {
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        fprintf(stderr, "[ptts] FlowNet input_proj: %.3f ms\n", ms);
    }

    if (profile) cudaEventRecord(ev_start, 0);
    if (linear_device(d_tmp, d_cond, desc->cond_w, desc->cond_b, FLOWLM_D_MODEL, FLOWLM_FLOW_DIM) != 0) return -1;
    if (add3_device(d_tmp2, d_ts, d_tt, d_tmp, FLOWLM_FLOW_DIM, 0.5f) != 0) return -1;
    if (profile) {
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        fprintf(stderr, "[ptts] FlowNet cond+time: %.3f ms\n", ms);
    }

    for (int b = 0; b < 6; b++) {
        const float *ln_w = desc->res[b].in_ln_w;
        const float *ln_b = desc->res[b].in_ln_b;
        if (profile) cudaEventRecord(ev_start, 0);
        if (layernorm_device(d_x, ln_w ? get_weight_device(ln_w, FLOWLM_FLOW_DIM * sizeof(float)) : NULL,
                             ln_b ? get_weight_device(ln_b, FLOWLM_FLOW_DIM * sizeof(float)) : NULL,
                             d_tmp, FLOWLM_FLOW_DIM, 1e-6f) != 0) return -1;

        if (cudaMemcpy(d_tmp3, d_tmp2, FLOWLM_FLOW_DIM * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) return -1;
        if (silu_device(d_tmp3, FLOWLM_FLOW_DIM) != 0) return -1;

        if (linear_device(d_ada, d_tmp3, desc->res[b].ada_w, desc->res[b].ada_b,
                          FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM * 3) != 0) return -1;
        if (profile) {
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            fprintf(stderr, "[ptts] FlowNet res%d norm+ada: %.3f ms\n", b, ms);
        }

        const float *d_shift = d_ada;
        const float *d_scale = d_ada + FLOWLM_FLOW_DIM;
        const float *d_gate = d_ada + 2 * FLOWLM_FLOW_DIM;
        if (affine_device(d_tmp, d_scale, d_shift, FLOWLM_FLOW_DIM) != 0) return -1;

        if (profile) cudaEventRecord(ev_start, 0);
        if (linear_device(d_mlp, d_tmp, desc->res[b].mlp0_w, desc->res[b].mlp0_b,
                          FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM) != 0) return -1;
        if (silu_device(d_mlp, FLOWLM_FLOW_DIM) != 0) return -1;
        if (linear_device(d_tmp3, d_mlp, desc->res[b].mlp2_w, desc->res[b].mlp2_b,
                          FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM) != 0) return -1;
        if (gate_add_device(d_x, d_gate, d_tmp3, FLOWLM_FLOW_DIM) != 0) return -1;
        if (profile) {
            cudaEventRecord(ev_stop, 0);
            cudaEventSynchronize(ev_stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            fprintf(stderr, "[ptts] FlowNet res%d mlp+gate: %.3f ms\n", b, ms);
        }
    }

    if (profile) cudaEventRecord(ev_start, 0);
    if (layernorm_device(d_x, NULL, NULL, d_tmp, FLOWLM_FLOW_DIM, 1e-6f) != 0) return -1;
    if (cudaMemcpy(d_tmp3, d_tmp2, FLOWLM_FLOW_DIM * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) return -1;
    if (silu_device(d_tmp3, FLOWLM_FLOW_DIM) != 0) return -1;
    if (linear_device(d_ada2, d_tmp3, desc->final.ada_w, desc->final.ada_b,
                      FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM * 2) != 0) return -1;
    const float *d_shift2 = d_ada2;
    const float *d_scale2 = d_ada2 + FLOWLM_FLOW_DIM;
    if (affine_device(d_tmp, d_scale2, d_shift2, FLOWLM_FLOW_DIM) != 0) return -1;
    if (linear_device(d_xin, d_tmp, desc->final.linear_w, desc->final.linear_b,
                      FLOWLM_FLOW_DIM, FLOWLM_LATENT_DIM) != 0) return -1;
    if (profile) {
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        fprintf(stderr, "[ptts] FlowNet final: %.3f ms\n", ms);
    }

    if (cudaMemcpy(out, d_xin, FLOWLM_LATENT_DIM * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (profile) {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }
    return 0;
}

int ptts_cuda_attention_forward(const float *q, const float *k, const float *v,
                                int T, int H, int D, float *out) {
    if (!q || !k || !v || !out || T <= 0 || H <= 0 || D <= 0) return -1;
    if (ensure_kernels() != 0) return -1;
    if (cuda_debug_enabled()) {
        fprintf(stderr, "[ptts] CUDA attention: T=%d H=%d D=%d\n", T, H, D);
    }

    size_t q_bytes = (size_t)T * H * D * sizeof(float);
    size_t out_bytes = (size_t)T * H * D * sizeof(float);

    if (ensure_device_buffer(&g_attn_q, &g_attn_q_bytes, q_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_k, &g_attn_k_bytes, q_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_v, &g_attn_v_bytes, q_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_out, &g_attn_out_bytes, out_bytes) != 0) return -1;

    if (cudaMemcpy(g_attn_q, q, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(g_attn_k, k, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(g_attn_v, v, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    float scale = 1.0f / sqrtf((float)D);
    int total = H * T;
    int block = 256;
    int grid = (total + block - 1) / block;
    void *args_fused[] = {&g_attn_q, &g_attn_k, &g_attn_v, &g_attn_out, &T, &H, &D, &scale};
    CUresult cuerr = cuLaunchKernel(g_attn_fused, grid, 1, 1, block, 1, 1, 0, 0, args_fused, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(attn_fused)", cuerr);
        return -1;
    }
    if (cuda_check("attn_fused") != 0) return -1;

    if (cudaMemcpy(out, g_attn_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

int ptts_cuda_attention_step(const float *q, const float *k, const float *v,
                             int T, int H, int D, float *out) {
    if (!q || !k || !v || !out || T <= 0 || H <= 0 || D <= 0) return -1;
    if (ensure_kernels() != 0) return -1;
    if (cuda_debug_enabled()) {
        fprintf(stderr, "[ptts] CUDA attention step: T=%d H=%d D=%d\n", T, H, D);
    }

    size_t q_bytes = (size_t)H * D * sizeof(float);
    size_t kv_bytes = (size_t)T * H * D * sizeof(float);
    size_t out_bytes = q_bytes;

    if (ensure_device_buffer(&g_attn_q, &g_attn_q_bytes, q_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_k, &g_attn_k_bytes, kv_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_v, &g_attn_v_bytes, kv_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_out, &g_attn_out_bytes, out_bytes) != 0) return -1;

    if (cudaMemcpy(g_attn_q, q, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(g_attn_k, k, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(g_attn_v, v, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    float scale = 1.0f / sqrtf((float)D);
    int block = 256;
    int grid = (H + block - 1) / block;
    void *args_step[] = {&g_attn_q, &g_attn_k, &g_attn_v, &g_attn_out, &T, &H, &D, &scale};
    CUresult cuerr = cuLaunchKernel(g_attn_step, grid, 1, 1, block, 1, 1, 0, 0, args_step, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(attn_step)", cuerr);
        return -1;
    }
    if (cuda_check("attn_step") != 0) return -1;

    if (cudaMemcpy(out, g_attn_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

int ptts_cuda_kv_init(int max_len) {
    if (max_len <= 0) return -1;
    if (ensure_kernels() != 0) return -1;
    return ensure_kv_cache(max_len);
}

int ptts_cuda_kv_push(int layer, int pos, const float *k, const float *v) {
    if (!k || !v || layer < 0 || layer >= FLOWLM_NUM_LAYERS || pos < 0) return -1;
    if (!g_k_cache_dev[layer] || !g_v_cache_dev[layer]) return -1;
    size_t stride = (size_t)FLOWLM_NUM_HEADS * FLOWLM_HEAD_DIM;
    size_t offset = (size_t)pos * stride;
    size_t bytes = stride * sizeof(float);
    float *dk = g_k_cache_dev[layer] + offset;
    float *dv = g_v_cache_dev[layer] + offset;
    if (cudaMemcpy(dk, k, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(dv, v, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    return 0;
}

int ptts_cuda_attention_step_kv(int layer, int T, int H, int D, const float *q, float *out) {
    if (!q || !out || T <= 0 || H <= 0 || D <= 0) return -1;
    if (layer < 0 || layer >= FLOWLM_NUM_LAYERS) return -1;
    if (!g_k_cache_dev[layer] || !g_v_cache_dev[layer]) return -1;
    if (ensure_kernels() != 0) return -1;

    size_t q_bytes = (size_t)H * D * sizeof(float);
    size_t out_bytes = q_bytes;
    if (ensure_device_buffer(&g_attn_q, &g_attn_q_bytes, q_bytes) != 0) return -1;
    if (ensure_device_buffer(&g_attn_out, &g_attn_out_bytes, out_bytes) != 0) return -1;

    if (cudaMemcpy(g_attn_q, q, q_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    float scale = 1.0f / sqrtf((float)D);
    int block = 256;
    int grid = (H + block - 1) / block;
    void *args_step[] = {&g_attn_q, &g_k_cache_dev[layer], &g_v_cache_dev[layer], &g_attn_out,
                         &T, &H, &D, &scale};
    CUresult cuerr = cuLaunchKernel(g_attn_step, grid, 1, 1, block, 1, 1, 0, 0, args_step, 0);
    if (cuerr != CUDA_SUCCESS) {
        cu_log_error("cuLaunchKernel(attn_step_kv)", cuerr);
        return -1;
    }
    if (cuda_check("attn_step_kv") != 0) return -1;
    if (cudaMemcpy(out, g_attn_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
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
    if (g_conv_buf3) cudaFree(g_conv_buf3);
    g_conv_buf0 = NULL;
    g_conv_buf1 = NULL;
    g_conv_buf2 = NULL;
    g_conv_buf3 = NULL;
    g_conv_buf0_bytes = 0;
    g_conv_buf1_bytes = 0;
    g_conv_buf2_bytes = 0;
    g_conv_buf3_bytes = 0;
    if (g_flow_buf0) cudaFree(g_flow_buf0);
    if (g_flow_buf1) cudaFree(g_flow_buf1);
    if (g_flow_buf2) cudaFree(g_flow_buf2);
    if (g_flow_buf3) cudaFree(g_flow_buf3);
    if (g_flow_buf4) cudaFree(g_flow_buf4);
    if (g_flow_buf5) cudaFree(g_flow_buf5);
    if (g_flow_buf6) cudaFree(g_flow_buf6);
    if (g_flow_buf7) cudaFree(g_flow_buf7);
    if (g_flow_buf8) cudaFree(g_flow_buf8);
    if (g_flow_buf9) cudaFree(g_flow_buf9);
    if (g_flow_buf10) cudaFree(g_flow_buf10);
    g_flow_buf0 = NULL;
    g_flow_buf1 = NULL;
    g_flow_buf2 = NULL;
    g_flow_buf3 = NULL;
    g_flow_buf4 = NULL;
    g_flow_buf5 = NULL;
    g_flow_buf6 = NULL;
    g_flow_buf7 = NULL;
    g_flow_buf8 = NULL;
    g_flow_buf9 = NULL;
    g_flow_buf10 = NULL;
    g_flow_buf0_bytes = 0;
    g_flow_buf1_bytes = 0;
    g_flow_buf2_bytes = 0;
    g_flow_buf3_bytes = 0;
    g_flow_buf4_bytes = 0;
    g_flow_buf5_bytes = 0;
    g_flow_buf6_bytes = 0;
    g_flow_buf7_bytes = 0;
    g_flow_buf8_bytes = 0;
    g_flow_buf9_bytes = 0;
    g_flow_buf10_bytes = 0;
    if (g_attn_q) cudaFree(g_attn_q);
    if (g_attn_k) cudaFree(g_attn_k);
    if (g_attn_v) cudaFree(g_attn_v);
    if (g_attn_scores_buf) cudaFree(g_attn_scores_buf);
    if (g_attn_out) cudaFree(g_attn_out);
    g_attn_q = NULL;
    g_attn_k = NULL;
    g_attn_v = NULL;
    g_attn_scores_buf = NULL;
    g_attn_out = NULL;
    g_attn_q_bytes = 0;
    g_attn_k_bytes = 0;
    g_attn_v_bytes = 0;
    g_attn_scores_bytes = 0;
    g_attn_out_bytes = 0;
    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        if (g_k_cache_dev[i]) cudaFree(g_k_cache_dev[i]);
        if (g_v_cache_dev[i]) cudaFree(g_v_cache_dev[i]);
        g_k_cache_dev[i] = NULL;
        g_v_cache_dev[i] = NULL;
    }
    g_kv_cache_bytes = 0;
    g_kv_cache_max_len = 0;
    cublasDestroy(g_handle);
    g_inited = 0;
}
