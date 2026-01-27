#include "ptts_mimi.h"
#include "ptts_internal.h"
#include "ptts_kernels.h"
#ifdef PTTS_USE_CUDA
#include "ptts_cuda.h"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIMI_D_MODEL 512
#define MIMI_NUM_HEADS 8
#define MIMI_HEAD_DIM 64
#define MIMI_NUM_LAYERS 2
#define MIMI_HIDDEN 2048
#define MIMI_CONTEXT 250

typedef struct {
    float *in_proj_w;
    float *out_proj_w;
    float *norm1_w;
    float *norm1_b;
    float *norm2_w;
    float *norm2_b;
    float *linear1_w;
    float *linear2_w;
    float *ls1; /* layer scale 1 */
    float *ls2; /* layer scale 2 */
} ptts_mimi_layer;

typedef struct {
    float *w;
    float *b;
    int out_ch;
    int in_ch;
    int k;
    int stride;
    int groups;
} ptts_conv1d;

typedef struct {
    float *w;
    float *b;
    int in_ch;
    int out_ch;
    int k;
    int stride;
    int groups;
} ptts_convtr1d;

typedef struct {
    ptts_conv1d conv1;
    ptts_conv1d conv2;
    int dim;
    int compress;
} ptts_resblock;

struct ptts_mimi {
    ptts_ctx *ctx;
    float *quant_w; /* [512, 32, 1] */
    ptts_convtr1d upsample;
    ptts_conv1d dec_in;
    ptts_convtr1d up[3];
    ptts_resblock res[3];
    ptts_conv1d dec_out;
    ptts_mimi_layer layers[MIMI_NUM_LAYERS];
};

static int ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t tlen = strlen(suffix);
    if (slen < tlen) return 0;
    return strcmp(s + slen - tlen, suffix) == 0;
}

static const safetensor_t *find_tensor_mimi(const ptts_ctx *ctx, const char *name) {
    const safetensor_t *t = safetensors_find(ctx->weights, name);
    if (t) return t;

    char buf[512];
    snprintf(buf, sizeof(buf), "mimi.%s", name);
    t = safetensors_find(ctx->weights, buf);
    if (t) return t;

    snprintf(buf, sizeof(buf), "model.%s", name);
    t = safetensors_find(ctx->weights, buf);
    if (t) return t;

    for (int i = 0; i < ctx->weights->num_tensors; i++) {
        const safetensor_t *cand = &ctx->weights->tensors[i];
        if (ends_with(cand->name, name)) return cand;
    }
    return NULL;
}

static float *load_f32(const ptts_ctx *ctx, const char *name) {
    const safetensor_t *t = find_tensor_mimi(ctx, name);
    if (!t) {
        fprintf(stderr, "Missing tensor: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(ctx->weights, t);
}

static float *load_f32_optional(const ptts_ctx *ctx, const char *name) {
    const safetensor_t *t = find_tensor_mimi(ctx, name);
    if (!t) return NULL;
    return safetensors_get_f32(ctx->weights, t);
}

static void free_ptr(float **p) {
    if (*p) free(*p);
    *p = NULL;
}

static void conv1d_forward_stream(const ptts_conv1d *c, const float *x, int T, float *y) {
    ptts_conv1d_forward(y, x, c->w, c->b, c->in_ch, c->out_ch, T, c->k, c->stride, c->groups);
}

#ifdef PTTS_USE_CUDA
static int cuda_conv_enabled(void) {
    static int inited = 0;
    static int enabled = 1;
    if (!inited) {
        const char *v = getenv("PTTS_CUDA_CONV");
        enabled = !(v && v[0] && strcmp(v, "0") == 0);
        inited = 1;
    }
    return enabled;
}
#endif

static void chw_to_thw(const float *in, int C, int T, float *out) {
    for (int t = 0; t < T; t++) {
        float *row = out + (size_t)t * C;
        for (int c = 0; c < C; c++) {
            row[c] = in[(size_t)c * T + t];
        }
    }
}

static void thw_to_chw(const float *in, int T, int C, float *out) {
    for (int c = 0; c < C; c++) {
        float *col = out + (size_t)c * T;
        for (int t = 0; t < T; t++) {
            col[t] = in[(size_t)t * C + c];
        }
    }
}

static void convtr1d_forward_stream(const ptts_convtr1d *c, const float *x, int T, float *y) {
    ptts_convtr1d_forward(y, x, c->w, c->b, c->in_ch, c->out_ch, T, c->k, c->stride, c->groups);
}

static void elu_inplace(float *x, int n) {
    ptts_elu_inplace(x, n);
}

static void resblock_forward(const ptts_resblock *rb, float *x, int T) {
    int dim = rb->dim;
    int out_len = T;
    int c1_out = rb->conv1.out_ch;
    float *tmp = (float *)malloc((size_t)dim * out_len * sizeof(float));
    float *tmp2 = (float *)malloc((size_t)c1_out * out_len * sizeof(float));
    if (!tmp || !tmp2) { free(tmp); free(tmp2); return; }

    memcpy(tmp, x, (size_t)dim * out_len * sizeof(float));
    elu_inplace(tmp, dim * out_len);
    conv1d_forward_stream(&rb->conv1, tmp, T, tmp2);
    elu_inplace(tmp2, c1_out * out_len);
    conv1d_forward_stream(&rb->conv2, tmp2, T, tmp);

    ptts_add_inplace(x, tmp, dim * out_len);

    free(tmp); free(tmp2);
}

static void linear_forward(const float *w, const float *b, int out, int in,
                           const float *x, int n, float *y) {
    ptts_linear_forward(y, x, w, b, n, in, out);
}

static void layernorm_forward(const float *x, int n, int d,
                              const float *w, const float *b, float eps, float *y) {
    for (int t = 0; t < n; t++) {
        const float *row = x + t * d;
        float *out = y + t * d;
        float mean = 0.0f;
        for (int i = 0; i < d; i++) mean += row[i];
        mean /= (float)d;
        float var = 0.0f;
        for (int i = 0; i < d; i++) {
            float v = row[i] - mean;
            var += v * v;
        }
        var /= (float)d;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < d; i++) {
            float v = (row[i] - mean) * inv;
            if (w) v *= w[i];
            if (b) v += b[i];
            out[i] = v;
        }
    }
}

static float gelu(float x) {
    const float k = 0.7978845608f; /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static void gelu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = gelu(x[i]);
}

static void softmax_inplace(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum == 0.0f) return;
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void rope_apply(float *q, float *k, int T, int H, int D, float max_period, int offset) {
    int half = D / 2;
    float *freqs = (float *)malloc((size_t)half * sizeof(float));
    if (!freqs) return;
    float log_mp = logf(max_period);
    for (int i = 0; i < half; i++) {
        freqs[i] = expf(-log_mp * (2.0f * i / D));
    }

    for (int t = 0; t < T; t++) {
        float ts = (float)(t + offset);
        for (int h = 0; h < H; h++) {
            float *qvec = q + (t * H + h) * D;
            float *kvec = k + (t * H + h) * D;
            for (int i = 0; i < half; i++) {
                float angle = freqs[i] * ts;
                float c = cosf(angle);
                float s = sinf(angle);
                int i0 = 2 * i;
                int i1 = i0 + 1;
                float qr = qvec[i0];
                float qi = qvec[i1];
                float kr = kvec[i0];
                float ki = kvec[i1];
                qvec[i0] = qr * c - qi * s;
                qvec[i1] = qr * s + qi * c;
                kvec[i0] = kr * c - ki * s;
                kvec[i1] = kr * s + ki * c;
            }
        }
    }
    free(freqs);
}

static void attention_forward_context(const float *q, const float *k, const float *v,
                                      int T, int H, int D, int context, float *out) {
    float scale = 1.0f / sqrtf((float)D);
    float *scores = (float *)malloc((size_t)T * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < H; h++) {
        for (int tq = 0; tq < T; tq++) {
            int n_keys = tq + 1;
            const float *qvec = q + (tq * H + h) * D;
            for (int tk = 0; tk < n_keys; tk++) {
                if (context > 0 && (tq - tk) >= context) {
                    scores[tk] = -1e30f;
                    continue;
                }
                const float *kvec = k + (tk * H + h) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qvec[d] * kvec[d];
                scores[tk] = dot * scale;
            }
            softmax_inplace(scores, n_keys);
            float *outvec = out + (tq * H + h) * D;
            for (int d = 0; d < D; d++) outvec[d] = 0.0f;
            for (int tk = 0; tk < n_keys; tk++) {
                const float *vvec = v + (tk * H + h) * D;
                float w = scores[tk];
                for (int d = 0; d < D; d++) outvec[d] += w * vvec[d];
            }
        }
    }
    free(scores);
}

static int transformer_forward(const ptts_mimi *mm, float *x, int T) {
    int d = MIMI_D_MODEL;
    int h = MIMI_NUM_HEADS;
    int hd = MIMI_HEAD_DIM;
    double t_start = 0.0;
    if (ptts_timing_enabled()) t_start = ptts_time_ms();

    float *x_norm = (float *)malloc((size_t)T * d * sizeof(float));
    float *qkv = (float *)malloc((size_t)T * d * 3 * sizeof(float));
    float *q = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *k = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *v = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *attn = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *attn_out = (float *)malloc((size_t)T * d * sizeof(float));
    float *ff1 = (float *)malloc((size_t)T * MIMI_HIDDEN * sizeof(float));
    float *ff2 = (float *)malloc((size_t)T * d * sizeof(float));

    if (!x_norm || !qkv || !q || !k || !v || !attn || !attn_out || !ff1 || !ff2) {
        free(x_norm); free(qkv); free(q); free(k); free(v); free(attn); free(attn_out); free(ff1); free(ff2);
        return -1;
    }

    for (int l = 0; l < MIMI_NUM_LAYERS; l++) {
        const ptts_mimi_layer *layer = &mm->layers[l];

        layernorm_forward(x, T, d, layer->norm1_w, layer->norm1_b, 1e-5f, x_norm);
        linear_forward(layer->in_proj_w, NULL, 3 * d, d, x_norm, T, qkv);

        for (int t = 0; t < T; t++) {
            const float *row = qkv + t * 3 * d;
            for (int hh = 0; hh < h; hh++) {
                float *qv = q + (t * h + hh) * hd;
                float *kv = k + (t * h + hh) * hd;
                float *vv = v + (t * h + hh) * hd;
                memcpy(qv, row + (0 * d + hh * hd), (size_t)hd * sizeof(float));
                memcpy(kv, row + (1 * d + hh * hd), (size_t)hd * sizeof(float));
                memcpy(vv, row + (2 * d + hh * hd), (size_t)hd * sizeof(float));
            }
        }

        rope_apply(q, k, T, h, hd, 10000.0f, 0);
        attention_forward_context(q, k, v, T, h, hd, MIMI_CONTEXT, attn);

        for (int t = 0; t < T; t++) {
            float *row = attn_out + t * d;
            for (int hh = 0; hh < h; hh++) {
                memcpy(row + hh * hd, attn + (t * h + hh) * hd, (size_t)hd * sizeof(float));
            }
        }

        linear_forward(layer->out_proj_w, NULL, d, d, attn_out, T, x_norm);
        for (int i = 0; i < T * d; i++) {
            float add = x_norm[i];
            if (layer->ls1) add *= layer->ls1[i % d];
            x[i] += add;
        }

        layernorm_forward(x, T, d, layer->norm2_w, layer->norm2_b, 1e-5f, x_norm);
        linear_forward(layer->linear1_w, NULL, MIMI_HIDDEN, d, x_norm, T, ff1);
        gelu_inplace(ff1, T * MIMI_HIDDEN);
        linear_forward(layer->linear2_w, NULL, d, MIMI_HIDDEN, ff1, T, ff2);
        for (int i = 0; i < T * d; i++) {
            float add = ff2[i];
            if (layer->ls2) add *= layer->ls2[i % d];
            x[i] += add;
        }
    }

    if (ptts_timing_enabled()) {
        double t_end = ptts_time_ms();
        fprintf(stderr, "[ptts] Mimi transformer: %.2f ms (T=%d)\n", t_end - t_start, T);
    }

    free(x_norm); free(qkv); free(q); free(k); free(v); free(attn); free(attn_out); free(ff1); free(ff2);
    return 0;
}

ptts_mimi *ptts_mimi_load(ptts_ctx *ctx) {
    if (!ctx || !ctx->weights) return NULL;
    ptts_mimi *mm = (ptts_mimi *)calloc(1, sizeof(ptts_mimi));
    if (!mm) return NULL;
    mm->ctx = ctx;

    mm->quant_w = load_f32(ctx, "quantizer.output_proj.weight");
    mm->upsample.w = load_f32_optional(ctx, "upsample.convtr.weight");
    if (!mm->upsample.w) {
        mm->upsample.w = load_f32(ctx, "upsample.convtr.convtr.weight");
    }
    mm->upsample.b = NULL;
    mm->upsample.in_ch = 512;
    mm->upsample.out_ch = 512;
    mm->upsample.k = 32;
    mm->upsample.stride = 16;
    mm->upsample.groups = 512;

    /* Decoder conv stack weights */
    mm->dec_in.w = load_f32(ctx, "decoder.model.0.conv.weight");
    mm->dec_in.b = load_f32(ctx, "decoder.model.0.conv.bias");
    mm->dec_in.in_ch = 512;
    mm->dec_in.out_ch = 512;
    mm->dec_in.k = 7;
    mm->dec_in.stride = 1;
    mm->dec_in.groups = 1;

    /* Stage 0: ratio 6 */
    mm->up[0].w = load_f32(ctx, "decoder.model.2.convtr.weight");
    mm->up[0].b = load_f32(ctx, "decoder.model.2.convtr.bias");
    mm->up[0].in_ch = 512;
    mm->up[0].out_ch = 256;
    mm->up[0].k = 12;
    mm->up[0].stride = 6;
    mm->up[0].groups = 1;
    mm->res[0].dim = 256;
    mm->res[0].compress = 2;
    mm->res[0].conv1.w = load_f32(ctx, "decoder.model.3.block.1.conv.weight");
    mm->res[0].conv1.b = load_f32(ctx, "decoder.model.3.block.1.conv.bias");
    mm->res[0].conv1.in_ch = 256;
    mm->res[0].conv1.out_ch = 128;
    mm->res[0].conv1.k = 3;
    mm->res[0].conv1.stride = 1;
    mm->res[0].conv1.groups = 1;
    mm->res[0].conv2.w = load_f32(ctx, "decoder.model.3.block.3.conv.weight");
    mm->res[0].conv2.b = load_f32(ctx, "decoder.model.3.block.3.conv.bias");
    mm->res[0].conv2.in_ch = 128;
    mm->res[0].conv2.out_ch = 256;
    mm->res[0].conv2.k = 1;
    mm->res[0].conv2.stride = 1;
    mm->res[0].conv2.groups = 1;

    /* Stage 1: ratio 5 */
    mm->up[1].w = load_f32(ctx, "decoder.model.5.convtr.weight");
    mm->up[1].b = load_f32(ctx, "decoder.model.5.convtr.bias");
    mm->up[1].in_ch = 256;
    mm->up[1].out_ch = 128;
    mm->up[1].k = 10;
    mm->up[1].stride = 5;
    mm->up[1].groups = 1;
    mm->res[1].dim = 128;
    mm->res[1].compress = 2;
    mm->res[1].conv1.w = load_f32(ctx, "decoder.model.6.block.1.conv.weight");
    mm->res[1].conv1.b = load_f32(ctx, "decoder.model.6.block.1.conv.bias");
    mm->res[1].conv1.in_ch = 128;
    mm->res[1].conv1.out_ch = 64;
    mm->res[1].conv1.k = 3;
    mm->res[1].conv1.stride = 1;
    mm->res[1].conv1.groups = 1;
    mm->res[1].conv2.w = load_f32(ctx, "decoder.model.6.block.3.conv.weight");
    mm->res[1].conv2.b = load_f32(ctx, "decoder.model.6.block.3.conv.bias");
    mm->res[1].conv2.in_ch = 64;
    mm->res[1].conv2.out_ch = 128;
    mm->res[1].conv2.k = 1;
    mm->res[1].conv2.stride = 1;
    mm->res[1].conv2.groups = 1;

    /* Stage 2: ratio 4 */
    mm->up[2].w = load_f32(ctx, "decoder.model.8.convtr.weight");
    mm->up[2].b = load_f32(ctx, "decoder.model.8.convtr.bias");
    mm->up[2].in_ch = 128;
    mm->up[2].out_ch = 64;
    mm->up[2].k = 8;
    mm->up[2].stride = 4;
    mm->up[2].groups = 1;
    mm->res[2].dim = 64;
    mm->res[2].compress = 2;
    mm->res[2].conv1.w = load_f32(ctx, "decoder.model.9.block.1.conv.weight");
    mm->res[2].conv1.b = load_f32(ctx, "decoder.model.9.block.1.conv.bias");
    mm->res[2].conv1.in_ch = 64;
    mm->res[2].conv1.out_ch = 32;
    mm->res[2].conv1.k = 3;
    mm->res[2].conv1.stride = 1;
    mm->res[2].conv1.groups = 1;
    mm->res[2].conv2.w = load_f32(ctx, "decoder.model.9.block.3.conv.weight");
    mm->res[2].conv2.b = load_f32(ctx, "decoder.model.9.block.3.conv.bias");
    mm->res[2].conv2.in_ch = 32;
    mm->res[2].conv2.out_ch = 64;
    mm->res[2].conv2.k = 1;
    mm->res[2].conv2.stride = 1;
    mm->res[2].conv2.groups = 1;

    mm->dec_out.w = load_f32(ctx, "decoder.model.11.conv.weight");
    mm->dec_out.b = load_f32(ctx, "decoder.model.11.conv.bias");
    mm->dec_out.in_ch = 64;
    mm->dec_out.out_ch = 1;
    mm->dec_out.k = 3;
    mm->dec_out.stride = 1;
    mm->dec_out.groups = 1;

    for (int i = 0; i < MIMI_NUM_LAYERS; i++) {
        char name[160];
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.self_attn.in_proj.weight", i);
        mm->layers[i].in_proj_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.self_attn.out_proj.weight", i);
        mm->layers[i].out_proj_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.norm1.weight", i);
        mm->layers[i].norm1_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.norm1.bias", i);
        mm->layers[i].norm1_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.norm2.weight", i);
        mm->layers[i].norm2_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.norm2.bias", i);
        mm->layers[i].norm2_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.linear1.weight", i);
        mm->layers[i].linear1_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.linear2.weight", i);
        mm->layers[i].linear2_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.layer_scale_1.scale", i);
        mm->layers[i].ls1 = load_f32(ctx, name);
        snprintf(name, sizeof(name), "decoder_transformer.transformer.layers.%d.layer_scale_2.scale", i);
        mm->layers[i].ls2 = load_f32(ctx, name);
    }

    if (!mm->quant_w || !mm->layers[0].in_proj_w || !mm->dec_out.w || !mm->upsample.w) {
        ptts_mimi_free(mm);
        return NULL;
    }
    return mm;
}

void ptts_mimi_free(ptts_mimi *mm) {
    if (!mm) return;
    free_ptr(&mm->quant_w);
    free_ptr(&mm->upsample.w);
    free_ptr(&mm->dec_in.w);
    free_ptr(&mm->dec_in.b);
    for (int i = 0; i < 3; i++) {
        free_ptr(&mm->up[i].w);
        free_ptr(&mm->up[i].b);
        free_ptr(&mm->res[i].conv1.w);
        free_ptr(&mm->res[i].conv1.b);
        free_ptr(&mm->res[i].conv2.w);
        free_ptr(&mm->res[i].conv2.b);
    }
    free_ptr(&mm->dec_out.w);
    free_ptr(&mm->dec_out.b);
    for (int i = 0; i < MIMI_NUM_LAYERS; i++) {
        free_ptr(&mm->layers[i].in_proj_w);
        free_ptr(&mm->layers[i].out_proj_w);
        free_ptr(&mm->layers[i].norm1_w);
        free_ptr(&mm->layers[i].norm1_b);
        free_ptr(&mm->layers[i].norm2_w);
        free_ptr(&mm->layers[i].norm2_b);
        free_ptr(&mm->layers[i].linear1_w);
        free_ptr(&mm->layers[i].linear2_w);
        free_ptr(&mm->layers[i].ls1);
        free_ptr(&mm->layers[i].ls2);
    }
    free(mm);
}

int ptts_mimi_forward_one(ptts_mimi *mm, const float *latent, float *out_embed) {
    if (!mm || !latent || !out_embed) return -1;

    float x[MIMI_D_MODEL];
    for (int o = 0; o < MIMI_D_MODEL; o++) {
        const float *wrow = mm->quant_w + o * 32;
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) sum += wrow[i] * latent[i];
        x[o] = sum;
    }

    if (transformer_forward(mm, x, 1) != 0) return -1;
    memcpy(out_embed, x, sizeof(x));
    return 0;
}

int ptts_mimi_decode_one(ptts_mimi *mm, const float *latent, float *out_audio, int *out_len) {
    return ptts_mimi_decode(mm, latent, 1, out_audio, out_len);
}

int ptts_mimi_decode(ptts_mimi *mm, const float *latents, int frames,
                     float *out_audio, int *out_len) {
    if (!mm || !latents || !out_audio || !out_len || frames < 1) return -1;

    /* quantizer output proj: [frames,32] -> [512,frames] (channel-major) */
    float *q = (float *)malloc((size_t)MIMI_D_MODEL * frames * sizeof(float));
    if (!q) return -1;
    for (int o = 0; o < MIMI_D_MODEL; o++) {
        const float *wrow = mm->quant_w + o * 32;
        float *dst = q + (size_t)o * frames;
        for (int t = 0; t < frames; t++) {
            const float *lat = latents + (size_t)t * 32;
            float sum = 0.0f;
            for (int i = 0; i < 32; i++) sum += wrow[i] * lat[i];
            dst[t] = sum;
        }
    }

    /* upsample convtr (groups=512): input length frames -> frames*16 */
    int up_len = frames * 16;
    float *up = (float *)malloc((size_t)MIMI_D_MODEL * up_len * sizeof(float));
    if (!up) { free(q); return -1; }
    convtr1d_forward_stream(&mm->upsample, q, frames, up);
    free(q);

    /* convert to time-major for transformer */
    float *up_t = (float *)malloc((size_t)up_len * MIMI_D_MODEL * sizeof(float));
    if (!up_t) { free(up); return -1; }
    chw_to_thw(up, MIMI_D_MODEL, up_len, up_t);

    if (transformer_forward(mm, up_t, up_len) != 0) {
        free(up_t);
        free(up);
        return -1;
    }

    /* back to channel-major for conv stack */
    thw_to_chw(up_t, up_len, MIMI_D_MODEL, up);
    free(up_t);

    /* decoder conv stack */
 #ifdef PTTS_USE_CUDA
    if (cuda_conv_enabled()) {
        double t_start = 0.0;
        int timing = ptts_timing_enabled();
        ptts_cuda_conv1d_desc dec_in = {
            mm->dec_in.w, mm->dec_in.b, mm->dec_in.in_ch, mm->dec_in.out_ch,
            mm->dec_in.k, mm->dec_in.stride, mm->dec_in.groups};
        ptts_cuda_convtr1d_desc up0 = {
            mm->up[0].w, mm->up[0].b, mm->up[0].in_ch, mm->up[0].out_ch,
            mm->up[0].k, mm->up[0].stride, mm->up[0].groups};
        ptts_cuda_conv1d_desc res0_1 = {
            mm->res[0].conv1.w, mm->res[0].conv1.b, mm->res[0].conv1.in_ch,
            mm->res[0].conv1.out_ch, mm->res[0].conv1.k, mm->res[0].conv1.stride,
            mm->res[0].conv1.groups};
        ptts_cuda_conv1d_desc res0_2 = {
            mm->res[0].conv2.w, mm->res[0].conv2.b, mm->res[0].conv2.in_ch,
            mm->res[0].conv2.out_ch, mm->res[0].conv2.k, mm->res[0].conv2.stride,
            mm->res[0].conv2.groups};
        ptts_cuda_convtr1d_desc up1 = {
            mm->up[1].w, mm->up[1].b, mm->up[1].in_ch, mm->up[1].out_ch,
            mm->up[1].k, mm->up[1].stride, mm->up[1].groups};
        ptts_cuda_conv1d_desc res1_1 = {
            mm->res[1].conv1.w, mm->res[1].conv1.b, mm->res[1].conv1.in_ch,
            mm->res[1].conv1.out_ch, mm->res[1].conv1.k, mm->res[1].conv1.stride,
            mm->res[1].conv1.groups};
        ptts_cuda_conv1d_desc res1_2 = {
            mm->res[1].conv2.w, mm->res[1].conv2.b, mm->res[1].conv2.in_ch,
            mm->res[1].conv2.out_ch, mm->res[1].conv2.k, mm->res[1].conv2.stride,
            mm->res[1].conv2.groups};
        ptts_cuda_convtr1d_desc up2 = {
            mm->up[2].w, mm->up[2].b, mm->up[2].in_ch, mm->up[2].out_ch,
            mm->up[2].k, mm->up[2].stride, mm->up[2].groups};
        ptts_cuda_conv1d_desc res2_1 = {
            mm->res[2].conv1.w, mm->res[2].conv1.b, mm->res[2].conv1.in_ch,
            mm->res[2].conv1.out_ch, mm->res[2].conv1.k, mm->res[2].conv1.stride,
            mm->res[2].conv1.groups};
        ptts_cuda_conv1d_desc res2_2 = {
            mm->res[2].conv2.w, mm->res[2].conv2.b, mm->res[2].conv2.in_ch,
            mm->res[2].conv2.out_ch, mm->res[2].conv2.k, mm->res[2].conv2.stride,
            mm->res[2].conv2.groups};
        ptts_cuda_conv1d_desc dec_out = {
            mm->dec_out.w, mm->dec_out.b, mm->dec_out.in_ch, mm->dec_out.out_ch,
            mm->dec_out.k, mm->dec_out.stride, mm->dec_out.groups};
        int cuda_len = 0;
        if (timing) t_start = ptts_time_ms();
        if (ptts_cuda_mimi_convstack(&dec_in, &up0, &res0_1, &res0_2,
                                     &up1, &res1_1, &res1_2,
                                     &up2, &res2_1, &res2_2,
                                     &dec_out, up, up_len, out_audio, &cuda_len) == 0) {
            if (timing) {
                double t_end = ptts_time_ms();
                fprintf(stderr, "[ptts] Mimi conv stack (CUDA): %.2f ms\n", t_end - t_start);
            }
            *out_len = cuda_len;
            free(up);
            return 0;
        }
        if (timing) {
            double t_end = ptts_time_ms();
            fprintf(stderr, "[ptts] Mimi conv stack (CUDA) failed: %.2f ms, falling back to CPU\n",
                    t_end - t_start);
        }
    }
#endif

    float *x = up;
    int T = up_len;

    double t_cpu = 0.0;
    int timing = ptts_timing_enabled();
    if (timing) t_cpu = ptts_time_ms();

    float *tmp = (float *)malloc((size_t)mm->dec_in.out_ch * T * sizeof(float));
    if (!tmp) { free(up); return -1; }
    conv1d_forward_stream(&mm->dec_in, x, T, tmp);
    free(x);
    x = tmp;

    /* stage 0 */
    elu_inplace(x, mm->dec_in.out_ch * T);
    int t0 = T * 6;
    float *u0 = (float *)malloc((size_t)mm->up[0].out_ch * t0 * sizeof(float));
    if (!u0) { free(x); return -1; }
    convtr1d_forward_stream(&mm->up[0], x, T, u0);
    free(x);
    x = u0;
    T = t0;
    resblock_forward(&mm->res[0], x, T);

    /* stage 1 */
    elu_inplace(x, mm->res[0].dim * T);
    int t1 = T * 5;
    float *u1 = (float *)malloc((size_t)mm->up[1].out_ch * t1 * sizeof(float));
    if (!u1) { free(x); return -1; }
    convtr1d_forward_stream(&mm->up[1], x, T, u1);
    free(x);
    x = u1;
    T = t1;
    resblock_forward(&mm->res[1], x, T);

    /* stage 2 */
    elu_inplace(x, mm->res[1].dim * T);
    int t2 = T * 4;
    float *u2 = (float *)malloc((size_t)mm->up[2].out_ch * t2 * sizeof(float));
    if (!u2) { free(x); return -1; }
    convtr1d_forward_stream(&mm->up[2], x, T, u2);
    free(x);
    x = u2;
    T = t2;
    resblock_forward(&mm->res[2], x, T);

    /* final conv to mono */
    elu_inplace(x, mm->res[2].dim * T);
    float *out = (float *)malloc((size_t)mm->dec_out.out_ch * T * sizeof(float));
    if (!out) { free(x); return -1; }
    conv1d_forward_stream(&mm->dec_out, x, T, out);
    free(x);

    if (timing) {
        double t_end = ptts_time_ms();
        fprintf(stderr, "[ptts] Mimi conv stack (CPU): %.2f ms\n", t_end - t_cpu);
    }

    memcpy(out_audio, out, (size_t)T * sizeof(float));
    *out_len = T;
    free(out);
    return 0;
}
