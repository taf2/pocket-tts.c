#ifndef PTTS_MPS_H
#define PTTS_MPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize MPS backend - must be called before other functions
int ptts_mps_init(void);

// Cleanup MPS resources
void ptts_mps_cleanup(void);

// Check if MPS is available and initialized
int ptts_mps_available(void);

// Memory management
void ptts_mps_clear_weight_cache(void);
size_t ptts_mps_memory_used(void);

// Batch execution control
void ptts_mps_begin_batch(void);
void ptts_mps_end_batch(void);

// Core operations - return 0 on success, -1 on failure

// Linear: y = x @ W^T + b
// x: [n, in], w: [out, in], b: [out], y: [n, out]
int ptts_mps_linear_forward(float *y, const float *x, const float *w,
                            const float *b, int n, int in_features, int out_features);

// Conv1d: y = conv(x, w) + b
// x: [in_ch, T], w: [out_ch, in_ch/groups, k], b: [out_ch], y: [out_ch, out_len]
int ptts_mps_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                            int in_ch, int out_ch, int T, int k, int stride, int groups);

// ConvTranspose1d (deconvolution/upsampling)
int ptts_mps_convtr1d_forward(float *y, const float *x, const float *w, const float *b,
                              int in_ch, int out_ch, int T, int k, int stride, int groups);

// Attention: out = softmax(Q @ K^T / sqrt(d)) @ V
// Q: [n, heads, d], K: [m, heads, d], V: [m, heads, d], out: [n, heads, d]
int ptts_mps_attention_forward(float *out, const float *Q, const float *K, const float *V,
                               int n, int m, int heads, int d, int causal);

// Layer normalization
int ptts_mps_layernorm_forward(float *y, const float *x, const float *gamma,
                               const float *beta, int n, int d, float eps);

// RMS normalization
int ptts_mps_rmsnorm_forward(float *y, const float *x, const float *gamma,
                             int n, int d, float eps);

// Activation functions (in-place)
int ptts_mps_elu_forward(float *x, int n, float alpha);
int ptts_mps_silu_forward(float *x, int n);

// Fused operations
int ptts_mps_linear_silu_forward(float *y, const float *x, const float *w,
                                 const float *b, int n, int in_features, int out_features);

#ifdef __cplusplus
}
#endif

#endif // PTTS_MPS_H
