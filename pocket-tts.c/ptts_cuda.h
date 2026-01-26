#ifndef PTTS_CUDA_H
#define PTTS_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 on success, non-zero on failure. */
int ptts_cuda_linear_forward(float *y, const float *x, const float *w, const float *b,
                             int n, int in, int out);

int ptts_cuda_conv1d_forward(float *y, const float *x, const float *w, const float *b,
                             int in_ch, int out_ch, int T, int k, int stride, int groups);

int ptts_cuda_convtr1d_forward(float *y, const float *x, const float *w, const float *b,
                               int in_ch, int out_ch, int T, int k, int stride, int groups);

typedef struct {
    const float *w;
    const float *b;
    int in_ch;
    int out_ch;
    int k;
    int stride;
    int groups;
} ptts_cuda_conv1d_desc;

typedef struct {
    const float *w;
    const float *b;
    int in_ch;
    int out_ch;
    int k;
    int stride;
    int groups;
} ptts_cuda_convtr1d_desc;

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
                             float *out_host, int *out_len);

void ptts_cuda_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_CUDA_H */
