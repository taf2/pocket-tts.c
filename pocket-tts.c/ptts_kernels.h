#ifndef PTTS_KERNELS_H
#define PTTS_KERNELS_H

/* Minimal kernel abstraction for backend acceleration. */

/* Linear layer: y = x @ W^T + b
 * x: [n, in], W: [out, in], b: [out], y: [n, out]
 */
void ptts_linear_forward(float *y, const float *x, const float *w, const float *b,
                         int n, int in, int out);

/* Conv1d: x [in_ch, T], w [out_ch, in_ch/groups, k], y [out_ch, out_len] */
void ptts_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                         int in_ch, int out_ch, int T, int k, int stride, int groups);

/* ConvTranspose1d: x [in_ch, T], w [in_ch, out_ch/groups, k], y [out_ch, out_len] */
void ptts_convtr1d_forward(float *y, const float *x, const float *w, const float *b,
                           int in_ch, int out_ch, int T, int k, int stride, int groups);

void ptts_elu_inplace(float *x, int n);
void ptts_add_inplace(float *a, const float *b, int n);

#endif /* PTTS_KERNELS_H */
