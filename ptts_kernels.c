#include "ptts_kernels.h"

#ifdef PTTS_USE_CUDA
#include "ptts_cuda.h"
#endif

#ifdef PTTS_USE_BLAS
#include <cblas.h>
#endif
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef PTTS_USE_CUDA
static int g_cuda_linear_inited = 0;
static int g_cuda_linear_enabled = 1;
static int g_cuda_conv1d_inited = 0;
static int g_cuda_conv1d_enabled = 1;
static int g_cuda_convtr_inited = 0;
static int g_cuda_convtr_enabled = 1;

static int cuda_linear_enabled(void) {
    if (!g_cuda_linear_inited) {
        const char *v = getenv("PTTS_CUDA_LINEAR");
        g_cuda_linear_enabled = !(v && v[0] && strcmp(v, "0") == 0);
        g_cuda_linear_inited = 1;
    }
    return g_cuda_linear_enabled;
}

static int cuda_conv1d_enabled(void) {
    if (!g_cuda_conv1d_inited) {
        const char *v = getenv("PTTS_CUDA_CONV1D");
        g_cuda_conv1d_enabled = !(v && v[0] && strcmp(v, "0") == 0);
        g_cuda_conv1d_inited = 1;
    }
    return g_cuda_conv1d_enabled;
}

static int cuda_convtr_enabled(void) {
    if (!g_cuda_convtr_inited) {
        const char *v = getenv("PTTS_CUDA_CONVTR");
        g_cuda_convtr_enabled = !(v && v[0] && strcmp(v, "0") == 0);
        g_cuda_convtr_inited = 1;
    }
    return g_cuda_convtr_enabled;
}
#endif

void ptts_linear_forward(float *y, const float *x, const float *w, const float *b,
                         int n, int in, int out) {
#ifdef PTTS_USE_CUDA
    if (cuda_linear_enabled() && ptts_cuda_linear_forward(y, x, w, b, n, in, out) == 0) {
        return;
    }
#endif
#ifdef PTTS_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, out, in, 1.0f, x, in, w, in, 0.0f, y, out);
    if (b) {
        for (int t = 0; t < n; t++) {
            float *yrow = y + t * out;
            for (int o = 0; o < out; o++) yrow[o] += b[o];
        }
    }
#else
    for (int t = 0; t < n; t++) {
        const float *xrow = x + t * in;
        float *yrow = y + t * out;
        for (int o = 0; o < out; o++) {
            const float *wrow = w + o * in;
            float sum = b ? b[o] : 0.0f;
            for (int i = 0; i < in; i++) sum += wrow[i] * xrow[i];
            yrow[o] = sum;
        }
    }
#endif
}

void ptts_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                         int in_ch, int out_ch, int T, int k, int stride, int groups) {
#ifdef PTTS_USE_CUDA
    if (cuda_conv1d_enabled() &&
        ptts_cuda_conv1d_forward(y, x, w, b, in_ch, out_ch, T, k, stride, groups) == 0) {
        return;
    }
#endif
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

void ptts_convtr1d_forward(float *y, const float *x, const float *w, const float *b,
                           int in_ch, int out_ch, int T, int k, int stride, int groups) {
#ifdef PTTS_USE_CUDA
    if (cuda_convtr_enabled() &&
        ptts_cuda_convtr1d_forward(y, x, w, b, in_ch, out_ch, T, k, stride, groups) == 0) {
        return;
    }
#endif
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

void ptts_elu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v >= 0.0f ? v : (expf(v) - 1.0f);
    }
}

void ptts_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}
