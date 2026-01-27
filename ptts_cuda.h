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

typedef struct {
    const float *cond_w;
    const float *cond_b;
    const float *input_w;
    const float *input_b;
    struct {
        const float *lin0_w;
        const float *lin0_b;
        const float *lin2_w;
        const float *lin2_b;
        const float *rms_alpha;
        const float *freqs;
    } time[2];
    struct {
        const float *in_ln_w;
        const float *in_ln_b;
        const float *mlp0_w;
        const float *mlp0_b;
        const float *mlp2_w;
        const float *mlp2_b;
        const float *ada_w;
        const float *ada_b;
    } res[6];
    struct {
        const float *linear_w;
        const float *linear_b;
        const float *ada_w;
        const float *ada_b;
    } final;
} ptts_cuda_flow_net_desc;

int ptts_cuda_flownet_forward(const ptts_cuda_flow_net_desc *desc,
                              const float *cond, const float *ts, const float *tt,
                              const float *x_in, float *out);

int ptts_cuda_attention_forward(const float *q, const float *k, const float *v,
                                int T, int H, int D, float *out);
int ptts_cuda_attention_step(const float *q, const float *k, const float *v,
                             int T, int H, int D, float *out);
int ptts_cuda_kv_init(int max_len);
int ptts_cuda_kv_push(int layer, int pos, const float *k, const float *v);
int ptts_cuda_attention_step_kv(int layer, int T, int H, int D,
                                const float *q, float *out);

void ptts_cuda_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_CUDA_H */
