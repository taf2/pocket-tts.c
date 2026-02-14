#include "ptts_kernels.h"

#ifdef PTTS_USE_CUDA
#include "ptts_cuda.h"
#endif

#ifdef PTTS_USE_MPS
#include "ptts_mps.h"
static int g_mps_initialized = 0;
static int g_mps_enabled = 1;

static int mps_init_if_needed(void) {
    if (!g_mps_initialized) {
        if (ptts_mps_init() == 0) {
            g_mps_initialized = 1;
        }
    }
    return g_mps_initialized && g_mps_enabled;
}
#endif

#ifdef PTTS_USE_BLAS
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
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
#ifdef PTTS_USE_MPS
    if (mps_init_if_needed() && ptts_mps_linear_forward(y, x, w, b, n, in, out) == 0) {
        return;
    }
#endif
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
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int t = 0; t < n; t++) {
        for (int o = 0; o < out; o++) {
            const float *xrow = x + t * in;
            const float *wrow = w + o * in;
            float sum = b ? b[o] : 0.0f;
            int i = 0;
            for (; i <= in - 4; i += 4) {
                sum += wrow[i] * xrow[i] +
                       wrow[i+1] * xrow[i+1] +
                       wrow[i+2] * xrow[i+2] +
                       wrow[i+3] * xrow[i+3];
            }
            for (; i < in; i++) sum += wrow[i] * xrow[i];
            y[(size_t)t * out + o] = sum;
        }
    }
#endif
}

void ptts_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                         int in_ch, int out_ch, int T, int k, int stride, int groups) {
#ifdef PTTS_USE_MPS
    if (mps_init_if_needed() &&
        ptts_mps_conv1d_forward(y, x, w, b, in_ch, out_ch, T, k, stride, groups) == 0) {
        return;
    }
#endif
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

#ifdef _OPENMP
    #pragma omp parallel for
#endif
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
#ifdef PTTS_USE_MPS
    if (mps_init_if_needed() &&
        ptts_mps_convtr1d_forward(y, x, w, b, in_ch, out_ch, T, k, stride, groups) == 0) {
        return;
    }
#endif
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

    // Parallelize over output channels to avoid race conditions (no atomic needed)
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int oc = 0; oc < out_ch; oc++) {
        int g = oc / out_per_group;
        int ocg = oc % out_per_group;
        int in_base = g * in_per_group;
        float *ych = y + (size_t)oc * out_len;

        float bias = b ? b[oc] : 0.0f;
        for(int t=0; t<out_len; t++) ych[t] = bias;

        for (int ic_offset = 0; ic_offset < in_per_group; ic_offset++) {
            int ic = in_base + ic_offset;
            const float *xch = x + (size_t)ic * T;
            const float *wrow = w + ((size_t)ic * out_per_group + ocg) * k;

            for (int t = 0; t < T; t++) {
                int out_start = t * stride;
                float xval = xch[t];
                for (int kk = 0; kk < k; kk++) {
                   int idx = out_start + kk;
                   if (idx < out_len) {
                       ych[idx] += wrow[kk] * xval;
                   }
                }
            }
        }
    }
}

void ptts_elu_inplace(float *x, int n) {
#ifdef PTTS_USE_MPS
    if (mps_init_if_needed() && ptts_mps_elu_forward(x, n, 1.0f) == 0) {
        return;
    }
#endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v >= 0.0f ? v : (expf(v) - 1.0f);
    }
}

void ptts_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}
