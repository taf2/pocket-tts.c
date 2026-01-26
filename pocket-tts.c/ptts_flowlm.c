#include "ptts_flowlm.h"
#include "ptts_internal.h"
#include "ptts_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FLOWLM_VOCAB 4000
#define FLOWLM_TEXT_DIM 1024
#define FLOWLM_D_MODEL 1024
#define FLOWLM_NUM_HEADS 16
#define FLOWLM_HEAD_DIM 64
#define FLOWLM_NUM_LAYERS 6
#define FLOWLM_HIDDEN 4096
#define FLOWLM_LATENT_DIM 32
#define FLOWLM_FLOW_DIM 512
#define FLOWLM_FLOW_DEPTH 6
#define FLOWLM_MAX_PERIOD 10000.0f

typedef struct {
    float *in_proj_w;   /* [3*d_model, d_model] */
    float *out_proj_w;  /* [d_model, d_model] */
    float *norm1_w;
    float *norm1_b;
    float *norm2_w;
    float *norm2_b;
    float *linear1_w;   /* [hidden, d_model] */
    float *linear2_w;   /* [d_model, hidden] */
} ptts_flowlm_layer;

typedef struct {
    float *lin0_w; /* [flow_dim, 256] */
    float *lin0_b;
    float *lin2_w; /* [flow_dim, flow_dim] */
    float *lin2_b;
    float *rms_alpha; /* [flow_dim] */
    float *freqs;     /* [128] */
} ptts_time_embed;

typedef struct {
    float *in_ln_w;
    float *in_ln_b;
    float *mlp0_w;
    float *mlp0_b;
    float *mlp2_w;
    float *mlp2_b;
    float *ada_w; /* [3*flow_dim, flow_dim] */
    float *ada_b; /* [3*flow_dim] */
} ptts_resblock;

typedef struct {
    float *linear_w; /* [latent_dim, flow_dim] */
    float *linear_b;
    float *ada_w;    /* [2*flow_dim, flow_dim] */
    float *ada_b;    /* [2*flow_dim] */
} ptts_final_layer;

typedef struct {
    float *cond_w; /* [flow_dim, d_model] */
    float *cond_b;
    float *input_w; /* [flow_dim, latent_dim] */
    float *input_b;
    ptts_time_embed time[2];
    ptts_resblock res[FLOWLM_FLOW_DEPTH];
    ptts_final_layer final;
} ptts_flow_net;

struct ptts_flowlm {
    ptts_ctx *ctx;
    float *embed_weight; /* [vocab+1, text_dim] */
    float *speaker_proj; /* [text_dim, 512] */
    float *emb_std;      /* [latent_dim] */
    float *emb_mean;     /* [latent_dim] */
    float *bos_emb;      /* [latent_dim] */
    float *input_linear_w; /* [d_model, latent_dim] */
    float *out_norm_w;     /* [d_model] */
    float *out_norm_b;     /* [d_model] */
    float *out_eos_w;      /* [1, d_model] */
    float *out_eos_b;      /* [1] */
    ptts_flowlm_layer layers[FLOWLM_NUM_LAYERS];
    ptts_flow_net flow;
};

/* ========================================================================
 * Helpers
 * ======================================================================== */

static int ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t tlen = strlen(suffix);
    if (slen < tlen) return 0;
    return strcmp(s + slen - tlen, suffix) == 0;
}

static const safetensor_t *find_tensor_flowlm(const ptts_ctx *ctx, const char *name) {
    const safetensor_t *t = safetensors_find(ctx->weights, name);
    if (t) return t;

    char buf[512];
    snprintf(buf, sizeof(buf), "flow_lm.%s", name);
    t = safetensors_find(ctx->weights, buf);
    if (t) return t;

    /* fallback by suffix */
    for (int i = 0; i < ctx->weights->num_tensors; i++) {
        const safetensor_t *cand = &ctx->weights->tensors[i];
        if (ends_with(cand->name, name)) return cand;
    }
    return NULL;
}

static float *load_f32(const ptts_ctx *ctx, const char *name) {
    const safetensor_t *t = find_tensor_flowlm(ctx, name);
    if (!t) {
        fprintf(stderr, "Missing tensor: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(ctx->weights, t);
}

static void free_ptr(float **p) {
    if (*p) free(*p);
    *p = NULL;
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

static void rmsnorm_forward(const float *x, int d, const float *alpha, float eps, float *y) {
    float mean = 0.0f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= (float)d;
    float var = 0.0f;
    for (int i = 0; i < d; i++) {
        float v = x[i] - mean;
        var += v * v;
    }
    if (d > 1) var /= (float)(d - 1);
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < d; i++) {
        y[i] = x[i] * (alpha ? alpha[i] : 1.0f) * inv;
    }
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = silu(x[i]);
}

static float gelu(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
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

static void attention_forward(const float *q, const float *k, const float *v,
                              int T, int H, int D, float *out) {
    float scale = 1.0f / sqrtf((float)D);
    int max_keys = T;
    float *scores = (float *)malloc((size_t)max_keys * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < H; h++) {
        for (int tq = 0; tq < T; tq++) {
            int n_keys = tq + 1;
            const float *qvec = q + (tq * H + h) * D;
            for (int tk = 0; tk < n_keys; tk++) {
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

static void rope_apply_one(float *q, float *k, int H, int D, float max_period, int pos) {
    int half = D / 2;
    static int init = 0;
    static float freqs[FLOWLM_HEAD_DIM / 2];
    if (!init) {
        float log_mp = logf(max_period);
        for (int i = 0; i < half; i++) {
            freqs[i] = expf(-log_mp * (2.0f * i / D));
        }
        init = 1;
    }

    float ts = (float)pos;
    for (int h = 0; h < H; h++) {
        float *qvec = q + h * D;
        float *kvec = k + h * D;
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

typedef struct {
    int max_len;
    int seq_len;
    float *k_cache[FLOWLM_NUM_LAYERS];
    float *v_cache[FLOWLM_NUM_LAYERS];
    float *scores;
} ptts_flowlm_kv_cache;

static void kv_cache_free(ptts_flowlm_kv_cache *cache);

static ptts_flowlm_kv_cache *kv_cache_create(int max_len) {
    ptts_flowlm_kv_cache *cache = (ptts_flowlm_kv_cache *)calloc(1, sizeof(*cache));
    if (!cache) return NULL;
    cache->max_len = max_len;
    cache->seq_len = 0;
    size_t kv_elems = (size_t)max_len * FLOWLM_NUM_HEADS * FLOWLM_HEAD_DIM;
    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        cache->k_cache[i] = (float *)malloc(kv_elems * sizeof(float));
        cache->v_cache[i] = (float *)malloc(kv_elems * sizeof(float));
        if (!cache->k_cache[i] || !cache->v_cache[i]) {
            kv_cache_free(cache);
            return NULL;
        }
    }
    cache->scores = (float *)malloc((size_t)max_len * sizeof(float));
    if (!cache->scores) {
        kv_cache_free(cache);
        return NULL;
    }
    return cache;
}

static void kv_cache_free(ptts_flowlm_kv_cache *cache) {
    if (!cache) return;
    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        free(cache->k_cache[i]);
        free(cache->v_cache[i]);
    }
    free(cache->scores);
    free(cache);
}

static int transformer_forward_step_cached(const ptts_flowlm *fm, ptts_flowlm_kv_cache *cache,
                                           float *x) {
    if (!fm || !cache || !x) return -1;
    if (cache->seq_len >= cache->max_len) return -1;

    int d = FLOWLM_D_MODEL;
    int h = FLOWLM_NUM_HEADS;
    int hd = FLOWLM_HEAD_DIM;
    int pos = cache->seq_len;

    float x_norm[FLOWLM_D_MODEL];
    float qkv[FLOWLM_D_MODEL * 3];
    float q[FLOWLM_D_MODEL];
    float k[FLOWLM_D_MODEL];
    float v[FLOWLM_D_MODEL];
    float attn_out[FLOWLM_D_MODEL];
    float ff1[FLOWLM_HIDDEN];
    float ff2[FLOWLM_D_MODEL];

    for (int l = 0; l < FLOWLM_NUM_LAYERS; l++) {
        const ptts_flowlm_layer *layer = &fm->layers[l];

        layernorm_forward(x, 1, d, layer->norm1_w, layer->norm1_b, 1e-5f, x_norm);
        linear_forward(layer->in_proj_w, NULL, 3 * d, d, x_norm, 1, qkv);

        memcpy(q, qkv, (size_t)d * sizeof(float));
        memcpy(k, qkv + d, (size_t)d * sizeof(float));
        memcpy(v, qkv + 2 * d, (size_t)d * sizeof(float));

        rope_apply_one(q, k, h, hd, FLOWLM_MAX_PERIOD, pos);

        size_t base = (size_t)pos * h * hd;
        memcpy(cache->k_cache[l] + base, k, (size_t)d * sizeof(float));
        memcpy(cache->v_cache[l] + base, v, (size_t)d * sizeof(float));

        float *scores = cache->scores;
        for (int hh = 0; hh < h; hh++) {
            float *out = attn_out + hh * hd;
            int n_keys = pos + 1;
            const float *qvec = q + hh * hd;
            for (int tk = 0; tk < n_keys; tk++) {
                const float *kvec = cache->k_cache[l] + ((size_t)tk * h + hh) * hd;
                float dot = 0.0f;
                for (int d0 = 0; d0 < hd; d0++) dot += qvec[d0] * kvec[d0];
                scores[tk] = dot / sqrtf((float)hd);
            }
            softmax_inplace(scores, n_keys);
            for (int d0 = 0; d0 < hd; d0++) out[d0] = 0.0f;
            for (int tk = 0; tk < n_keys; tk++) {
                const float *vvec = cache->v_cache[l] + ((size_t)tk * h + hh) * hd;
                float w = scores[tk];
                for (int d0 = 0; d0 < hd; d0++) out[d0] += w * vvec[d0];
            }
        }

        linear_forward(layer->out_proj_w, NULL, d, d, attn_out, 1, x_norm);
        for (int i = 0; i < d; i++) x[i] += x_norm[i];

        layernorm_forward(x, 1, d, layer->norm2_w, layer->norm2_b, 1e-5f, x_norm);
        linear_forward(layer->linear1_w, NULL, FLOWLM_HIDDEN, d, x_norm, 1, ff1);
        gelu_inplace(ff1, FLOWLM_HIDDEN);
        linear_forward(layer->linear2_w, NULL, d, FLOWLM_HIDDEN, ff1, 1, ff2);
        for (int i = 0; i < d; i++) x[i] += ff2[i];
    }

    cache->seq_len++;
    return 0;
}

/* ========================================================================
 * Flow net
 * ======================================================================== */

static void timestep_embed(const ptts_time_embed *te, float t, float *out) {
    float emb[256];
    for (int i = 0; i < 128; i++) {
        float freq = te->freqs ? te->freqs[i] : expf(-logf(FLOWLM_MAX_PERIOD) * ((float)i / 128.0f));
        float angle = freq * t;
        emb[i] = cosf(angle);
        emb[i + 128] = sinf(angle);
    }

    float tmp[FLOWLM_FLOW_DIM];
    linear_forward(te->lin0_w, te->lin0_b, FLOWLM_FLOW_DIM, 256, emb, 1, tmp);
    silu_inplace(tmp, FLOWLM_FLOW_DIM);
    linear_forward(te->lin2_w, te->lin2_b, FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM, tmp, 1, out);
    rmsnorm_forward(out, FLOWLM_FLOW_DIM, te->rms_alpha, 1e-5f, out);
}

static void flow_net_forward(const ptts_flowlm *fm, const float *cond, float s, float t,
                             const float *x_in, float *out) {
    float x[FLOWLM_FLOW_DIM];
    float tmp[FLOWLM_FLOW_DIM];
    float tmp2[FLOWLM_FLOW_DIM];
    float mlp[FLOWLM_FLOW_DIM];
    float ada[FLOWLM_FLOW_DIM * 3];

    /* input projection */
    linear_forward(fm->flow.input_w, fm->flow.input_b, FLOWLM_FLOW_DIM, FLOWLM_LATENT_DIM, x_in, 1, x);

    /* time embeddings */
    float ts[FLOWLM_FLOW_DIM];
    float tt[FLOWLM_FLOW_DIM];
    timestep_embed(&fm->flow.time[0], s, ts);
    timestep_embed(&fm->flow.time[1], t, tt);

    /* cond embed */
    linear_forward(fm->flow.cond_w, fm->flow.cond_b, FLOWLM_FLOW_DIM, FLOWLM_D_MODEL, cond, 1, tmp);

    for (int i = 0; i < FLOWLM_FLOW_DIM; i++) {
        tmp2[i] = (ts[i] + tt[i]) * 0.5f + tmp[i];
    }

    /* res blocks */
    for (int b = 0; b < FLOWLM_FLOW_DEPTH; b++) {
        const ptts_resblock *rb = &fm->flow.res[b];
        layernorm_forward(x, 1, FLOWLM_FLOW_DIM, rb->in_ln_w, rb->in_ln_b, 1e-6f, tmp);

        /* adaLN modulation */
        float y[FLOWLM_FLOW_DIM];
        memcpy(y, tmp2, sizeof(y));
        silu_inplace(y, FLOWLM_FLOW_DIM);
        linear_forward(rb->ada_w, rb->ada_b, FLOWLM_FLOW_DIM * 3, FLOWLM_FLOW_DIM, y, 1, ada);
        float *shift = ada;
        float *scale = ada + FLOWLM_FLOW_DIM;
        float *gate = ada + 2 * FLOWLM_FLOW_DIM;

        for (int i = 0; i < FLOWLM_FLOW_DIM; i++) {
            tmp[i] = tmp[i] * (1.0f + scale[i]) + shift[i];
        }

        /* MLP */
        linear_forward(rb->mlp0_w, rb->mlp0_b, FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM, tmp, 1, mlp);
        silu_inplace(mlp, FLOWLM_FLOW_DIM);
        linear_forward(rb->mlp2_w, rb->mlp2_b, FLOWLM_FLOW_DIM, FLOWLM_FLOW_DIM, mlp, 1, tmp);

        for (int i = 0; i < FLOWLM_FLOW_DIM; i++) {
            x[i] = x[i] + gate[i] * tmp[i];
        }
    }

    /* final layer */
    layernorm_forward(x, 1, FLOWLM_FLOW_DIM, NULL, NULL, 1e-6f, tmp);
    float y[FLOWLM_FLOW_DIM];
    memcpy(y, tmp2, sizeof(y));
    silu_inplace(y, FLOWLM_FLOW_DIM);
    float ada2[FLOWLM_FLOW_DIM * 2];
    linear_forward(fm->flow.final.ada_w, fm->flow.final.ada_b, FLOWLM_FLOW_DIM * 2, FLOWLM_FLOW_DIM, y, 1, ada2);
    float *shift2 = ada2;
    float *scale2 = ada2 + FLOWLM_FLOW_DIM;
    for (int i = 0; i < FLOWLM_FLOW_DIM; i++) {
        tmp[i] = tmp[i] * (1.0f + scale2[i]) + shift2[i];
    }
    linear_forward(fm->flow.final.linear_w, fm->flow.final.linear_b, FLOWLM_LATENT_DIM, FLOWLM_FLOW_DIM, tmp, 1, out);
}

static void lsd_decode(const ptts_flowlm *fm, const float *cond, int num_steps, float *x,
                       float *out_first_flow) {
    if (num_steps <= 0) return;
    for (int i = 0; i < num_steps; i++) {
        float s = (float)i / (float)num_steps;
        float t = (float)(i + 1) / (float)num_steps;
        float flow[FLOWLM_LATENT_DIM];
        flow_net_forward(fm, cond, s, t, x, flow);
        if (i == 0 && out_first_flow) {
            memcpy(out_first_flow, flow, sizeof(flow));
        }
        for (int d = 0; d < FLOWLM_LATENT_DIM; d++) {
            x[d] += flow[d] / (float)num_steps;
        }
    }
}

/* ========================================================================
 * Transformer forward
 * ======================================================================== */

static int transformer_forward(const ptts_flowlm *fm, float *x, int T) {
    int d = FLOWLM_D_MODEL;
    int h = FLOWLM_NUM_HEADS;
    int hd = FLOWLM_HEAD_DIM;

    float *x_norm = (float *)malloc((size_t)T * d * sizeof(float));
    float *qkv = (float *)malloc((size_t)T * d * 3 * sizeof(float));
    float *q = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *k = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *v = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *attn = (float *)malloc((size_t)T * h * hd * sizeof(float));
    float *attn_out = (float *)malloc((size_t)T * d * sizeof(float));
    float *ff1 = (float *)malloc((size_t)T * FLOWLM_HIDDEN * sizeof(float));
    float *ff2 = (float *)malloc((size_t)T * d * sizeof(float));

    if (!x_norm || !qkv || !q || !k || !v || !attn || !attn_out || !ff1 || !ff2) {
        free(x_norm); free(qkv); free(q); free(k); free(v); free(attn); free(attn_out); free(ff1); free(ff2);
        return -1;
    }

    for (int l = 0; l < FLOWLM_NUM_LAYERS; l++) {
        const ptts_flowlm_layer *layer = &fm->layers[l];

        /* Norm1 */
        layernorm_forward(x, T, d, layer->norm1_w, layer->norm1_b, 1e-5f, x_norm);

        /* QKV */
        linear_forward(layer->in_proj_w, NULL, 3 * d, d, x_norm, T, qkv);

        /* split qkv into q,k,v (T,H,D) */
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

        /* Rope */
        rope_apply(q, k, T, h, hd, FLOWLM_MAX_PERIOD, 0);

        /* Attention */
        attention_forward(q, k, v, T, h, hd, attn);

        /* concat heads to attn_out */
        for (int t = 0; t < T; t++) {
            float *row = attn_out + t * d;
            for (int hh = 0; hh < h; hh++) {
                memcpy(row + hh * hd, attn + (t * h + hh) * hd, (size_t)hd * sizeof(float));
            }
        }

        /* out proj */
        linear_forward(layer->out_proj_w, NULL, d, d, attn_out, T, x_norm);

        /* residual */
        for (int i = 0; i < T * d; i++) x[i] += x_norm[i];

        /* Norm2 */
        layernorm_forward(x, T, d, layer->norm2_w, layer->norm2_b, 1e-5f, x_norm);

        /* FF */
        linear_forward(layer->linear1_w, NULL, FLOWLM_HIDDEN, d, x_norm, T, ff1);
        gelu_inplace(ff1, T * FLOWLM_HIDDEN);
        linear_forward(layer->linear2_w, NULL, d, FLOWLM_HIDDEN, ff1, T, ff2);

        for (int i = 0; i < T * d; i++) x[i] += ff2[i];
    }

    free(x_norm); free(qkv); free(q); free(k); free(v); free(attn); free(attn_out); free(ff1); free(ff2);
    return 0;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

ptts_flowlm *ptts_flowlm_load(ptts_ctx *ctx) {
    if (!ctx || !ctx->weights) return NULL;

    ptts_flowlm *fm = (ptts_flowlm *)calloc(1, sizeof(ptts_flowlm));
    if (!fm) return NULL;
    fm->ctx = ctx;

    fm->embed_weight = load_f32(ctx, "conditioner.embed.weight");
    fm->speaker_proj = load_f32(ctx, "speaker_proj_weight");
    fm->emb_std = load_f32(ctx, "emb_std");
    fm->emb_mean = load_f32(ctx, "emb_mean");
    fm->bos_emb = load_f32(ctx, "bos_emb");
    fm->input_linear_w = load_f32(ctx, "input_linear.weight");
    fm->out_norm_w = load_f32(ctx, "out_norm.weight");
    fm->out_norm_b = load_f32(ctx, "out_norm.bias");
    fm->out_eos_w = load_f32(ctx, "out_eos.weight");
    fm->out_eos_b = load_f32(ctx, "out_eos.bias");

    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        char name[128];
        snprintf(name, sizeof(name), "transformer.layers.%d.self_attn.in_proj.weight", i);
        fm->layers[i].in_proj_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.self_attn.out_proj.weight", i);
        fm->layers[i].out_proj_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.norm1.weight", i);
        fm->layers[i].norm1_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.norm1.bias", i);
        fm->layers[i].norm1_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.norm2.weight", i);
        fm->layers[i].norm2_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.norm2.bias", i);
        fm->layers[i].norm2_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.linear1.weight", i);
        fm->layers[i].linear1_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "transformer.layers.%d.linear2.weight", i);
        fm->layers[i].linear2_w = load_f32(ctx, name);
    }

    fm->flow.cond_w = load_f32(ctx, "flow_net.cond_embed.weight");
    fm->flow.cond_b = load_f32(ctx, "flow_net.cond_embed.bias");
    fm->flow.input_w = load_f32(ctx, "flow_net.input_proj.weight");
    fm->flow.input_b = load_f32(ctx, "flow_net.input_proj.bias");

    for (int t = 0; t < 2; t++) {
        char name[160];
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.0.weight", t);
        fm->flow.time[t].lin0_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.0.bias", t);
        fm->flow.time[t].lin0_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.2.weight", t);
        fm->flow.time[t].lin2_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.2.bias", t);
        fm->flow.time[t].lin2_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.3.alpha", t);
        fm->flow.time[t].rms_alpha = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.freqs", t);
        fm->flow.time[t].freqs = load_f32(ctx, name);
    }

    for (int i = 0; i < FLOWLM_FLOW_DEPTH; i++) {
        char name[200];
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.in_ln.weight", i);
        fm->flow.res[i].in_ln_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.in_ln.bias", i);
        fm->flow.res[i].in_ln_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.0.weight", i);
        fm->flow.res[i].mlp0_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.0.bias", i);
        fm->flow.res[i].mlp0_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.2.weight", i);
        fm->flow.res[i].mlp2_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.2.bias", i);
        fm->flow.res[i].mlp2_b = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.adaLN_modulation.1.weight", i);
        fm->flow.res[i].ada_w = load_f32(ctx, name);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.adaLN_modulation.1.bias", i);
        fm->flow.res[i].ada_b = load_f32(ctx, name);
    }

    fm->flow.final.linear_w = load_f32(ctx, "flow_net.final_layer.linear.weight");
    fm->flow.final.linear_b = load_f32(ctx, "flow_net.final_layer.linear.bias");
    fm->flow.final.ada_w = load_f32(ctx, "flow_net.final_layer.adaLN_modulation.1.weight");
    fm->flow.final.ada_b = load_f32(ctx, "flow_net.final_layer.adaLN_modulation.1.bias");

    /* basic validation */
    if (!fm->embed_weight || !fm->bos_emb || !fm->layers[0].in_proj_w || !fm->flow.cond_w) {
        ptts_flowlm_free(fm);
        return NULL;
    }

    return fm;
}

void ptts_flowlm_free(ptts_flowlm *fm) {
    if (!fm) return;
    free_ptr(&fm->embed_weight);
    free_ptr(&fm->speaker_proj);
    free_ptr(&fm->emb_std);
    free_ptr(&fm->emb_mean);
    free_ptr(&fm->bos_emb);
    free_ptr(&fm->input_linear_w);
    free_ptr(&fm->out_norm_w);
    free_ptr(&fm->out_norm_b);
    free_ptr(&fm->out_eos_w);
    free_ptr(&fm->out_eos_b);

    for (int i = 0; i < FLOWLM_NUM_LAYERS; i++) {
        free_ptr(&fm->layers[i].in_proj_w);
        free_ptr(&fm->layers[i].out_proj_w);
        free_ptr(&fm->layers[i].norm1_w);
        free_ptr(&fm->layers[i].norm1_b);
        free_ptr(&fm->layers[i].norm2_w);
        free_ptr(&fm->layers[i].norm2_b);
        free_ptr(&fm->layers[i].linear1_w);
        free_ptr(&fm->layers[i].linear2_w);
    }

    free_ptr(&fm->flow.cond_w);
    free_ptr(&fm->flow.cond_b);
    free_ptr(&fm->flow.input_w);
    free_ptr(&fm->flow.input_b);
    for (int t = 0; t < 2; t++) {
        free_ptr(&fm->flow.time[t].lin0_w);
        free_ptr(&fm->flow.time[t].lin0_b);
        free_ptr(&fm->flow.time[t].lin2_w);
        free_ptr(&fm->flow.time[t].lin2_b);
        free_ptr(&fm->flow.time[t].rms_alpha);
        free_ptr(&fm->flow.time[t].freqs);
    }
    for (int i = 0; i < FLOWLM_FLOW_DEPTH; i++) {
        free_ptr(&fm->flow.res[i].in_ln_w);
        free_ptr(&fm->flow.res[i].in_ln_b);
        free_ptr(&fm->flow.res[i].mlp0_w);
        free_ptr(&fm->flow.res[i].mlp0_b);
        free_ptr(&fm->flow.res[i].mlp2_w);
        free_ptr(&fm->flow.res[i].mlp2_b);
        free_ptr(&fm->flow.res[i].ada_w);
        free_ptr(&fm->flow.res[i].ada_b);
    }
    free_ptr(&fm->flow.final.linear_w);
    free_ptr(&fm->flow.final.linear_b);
    free_ptr(&fm->flow.final.ada_w);
    free_ptr(&fm->flow.final.ada_b);

    free(fm);
}

static uint32_t rng_next_u32(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

static float rng_next_f01(uint64_t *state) {
    uint32_t u = rng_next_u32(state);
    return (u + 1.0f) / 4294967296.0f;
}

int ptts_flowlm_forward_next(ptts_flowlm *fm, const int *tokens, int token_len,
                             const float *cond_prefix, int cond_len,
                             const float *prev_latents, int prev_len,
                             int lsd_steps, float temp, float noise_clamp,
                             int64_t *seed_io, float *out_latent, float *out_eos_logit) {
    if (!fm || !tokens || token_len <= 0 || !out_latent) return -1;
    if (cond_len < 0) return -1;
    if (cond_len > 0 && !cond_prefix) return -1;

    int seq_len = prev_len + 1; /* BOS + previous latents */
    int prefix_len = token_len + cond_len;
    int T = prefix_len + seq_len;
    float *x = (float *)malloc((size_t)T * FLOWLM_D_MODEL * sizeof(float));
    float *input_lat = (float *)malloc((size_t)FLOWLM_LATENT_DIM * sizeof(float));
    float *input_proj = (float *)malloc((size_t)FLOWLM_D_MODEL * sizeof(float));
    if (!x || !input_lat || !input_proj) {
        free(x); free(input_lat); free(input_proj);
        return -1;
    }

    /* audio conditioning prefix */
    for (int t = 0; t < cond_len; t++) {
        const float *src = cond_prefix + (size_t)t * FLOWLM_D_MODEL;
        memcpy(x + (size_t)t * FLOWLM_D_MODEL, src,
               (size_t)FLOWLM_D_MODEL * sizeof(float));
    }

    /* text embeddings */
    for (int t = 0; t < token_len; t++) {
        int id = tokens[t];
        if (id < 0 || id >= FLOWLM_VOCAB + 1) id = 0;
        const float *src = fm->embed_weight + (size_t)id * FLOWLM_TEXT_DIM;
        memcpy(x + (size_t)(cond_len + t) * FLOWLM_D_MODEL, src,
               (size_t)FLOWLM_D_MODEL * sizeof(float));
    }

    /* BOS */
    memcpy(input_lat, fm->bos_emb, (size_t)FLOWLM_LATENT_DIM * sizeof(float));
    linear_forward(fm->input_linear_w, NULL, FLOWLM_D_MODEL, FLOWLM_LATENT_DIM, input_lat, 1, input_proj);
    memcpy(x + (size_t)prefix_len * FLOWLM_D_MODEL, input_proj, (size_t)FLOWLM_D_MODEL * sizeof(float));

    /* previous latents */
    for (int i = 0; i < prev_len; i++) {
        const float *lat = prev_latents + (size_t)i * FLOWLM_LATENT_DIM;
        linear_forward(fm->input_linear_w, NULL, FLOWLM_D_MODEL, FLOWLM_LATENT_DIM, lat, 1, input_proj);
        memcpy(x + (size_t)(prefix_len + 1 + i) * FLOWLM_D_MODEL, input_proj,
               (size_t)FLOWLM_D_MODEL * sizeof(float));
    }

    if (transformer_forward(fm, x, T) != 0) {
        free(x); free(input_lat); free(input_proj);
        return -1;
    }

    /* take last token */
    float *last = x + (size_t)(T - 1) * FLOWLM_D_MODEL;
    float normed[FLOWLM_D_MODEL];
    layernorm_forward(last, 1, FLOWLM_D_MODEL, fm->out_norm_w, fm->out_norm_b, 1e-5f, normed);

    float eos = 0.0f;
    for (int i = 0; i < FLOWLM_D_MODEL; i++) {
        eos += fm->out_eos_w[i] * normed[i];
    }
    eos += fm->out_eos_b ? fm->out_eos_b[0] : 0.0f;
    if (out_eos_logit) *out_eos_logit = eos;

    /* initialize latent with noise and decode */
    float latent[FLOWLM_LATENT_DIM];
    int64_t seed = seed_io ? *seed_io : -1;
    if (seed == -1) seed = (int64_t)time(NULL);
    uint64_t rng = (uint64_t)seed;
    float std = (temp > 0.0f) ? sqrtf(temp) : 0.0f;
    for (int i = 0; i < FLOWLM_LATENT_DIM; i += 2) {
        float z0 = 0.0f;
        float z1 = 0.0f;
        if (std > 0.0f) {
            float u1 = rng_next_f01(&rng);
            float u2 = rng_next_f01(&rng);
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * (float)M_PI * u2;
            z0 = r * cosf(theta) * std;
            z1 = r * sinf(theta) * std;
        }
        if (noise_clamp > 0.0f) {
            if (z0 < -noise_clamp) z0 = -noise_clamp;
            if (z0 > noise_clamp) z0 = noise_clamp;
            if (z1 < -noise_clamp) z1 = -noise_clamp;
            if (z1 > noise_clamp) z1 = noise_clamp;
        }
        latent[i] = z0;
        if (i + 1 < FLOWLM_LATENT_DIM) latent[i + 1] = z1;
    }
    lsd_decode(fm, normed, lsd_steps, latent, NULL);

    memcpy(out_latent, latent, sizeof(latent));

    if (seed_io) *seed_io = (int64_t)rng;

    free(x); free(input_lat); free(input_proj);
    return 0;
}

int ptts_flowlm_forward_one(ptts_flowlm *fm, const int *tokens, int token_len,
                            const float *cond_prefix, int cond_len,
                            int lsd_steps, float temp, float noise_clamp,
                            int64_t seed, float *out_latent, float *out_eos_logit) {
    return ptts_flowlm_forward_next(fm, tokens, token_len, cond_prefix, cond_len,
                                    NULL, 0, lsd_steps, temp, noise_clamp,
                                    &seed, out_latent, out_eos_logit);
}

int ptts_flowlm_generate_latents(ptts_flowlm *fm, const int *tokens, int token_len,
                                 const float *cond_prefix, int cond_len,
                                 int max_frames, int lsd_steps, float temp, float noise_clamp,
                                 int64_t seed, int eos_enabled, float eos_threshold,
                                 int eos_min_frames, int eos_after,
                                 float *out_latents, int *out_frames_used,
                                 float *out_first_eos_logit,
                                 float *out_first_cond,
                                 float *out_first_flow) {
    if (!fm || !tokens || token_len <= 0 || !out_latents || !out_frames_used) return -1;
    if (max_frames < 1) return -1;
    if (cond_len < 0) return -1;
    if (cond_len > 0 && !cond_prefix) return -1;
    if (eos_min_frames < 1) eos_min_frames = 1;
    if (eos_after < 0) eos_after = 0;

    int max_len = token_len + cond_len + 1 + max_frames;
    ptts_flowlm_kv_cache *cache = kv_cache_create(max_len);
    if (!cache) return -1;

    float x[FLOWLM_D_MODEL];
    for (int t = 0; t < cond_len; t++) {
        const float *src = cond_prefix + (size_t)t * FLOWLM_D_MODEL;
        memcpy(x, src, (size_t)FLOWLM_D_MODEL * sizeof(float));
        if (transformer_forward_step_cached(fm, cache, x) != 0) {
            kv_cache_free(cache);
            return -1;
        }
    }

    for (int t = 0; t < token_len; t++) {
        int id = tokens[t];
        if (id < 0 || id >= FLOWLM_VOCAB + 1) id = 0;
        const float *src = fm->embed_weight + (size_t)id * FLOWLM_TEXT_DIM;
        memcpy(x, src, (size_t)FLOWLM_D_MODEL * sizeof(float));
        if (transformer_forward_step_cached(fm, cache, x) != 0) {
            kv_cache_free(cache);
            return -1;
        }
    }

    float input_lat[FLOWLM_LATENT_DIM];
    memcpy(input_lat, fm->bos_emb, (size_t)FLOWLM_LATENT_DIM * sizeof(float));
    linear_forward(fm->input_linear_w, NULL, FLOWLM_D_MODEL, FLOWLM_LATENT_DIM, input_lat, 1, x);
    if (transformer_forward_step_cached(fm, cache, x) != 0) {
        kv_cache_free(cache);
        return -1;
    }

    int64_t seed_local = seed;
    if (seed_local == -1) seed_local = (int64_t)time(NULL);
    uint64_t rng = (uint64_t)seed_local;
    float std = (temp > 0.0f) ? sqrtf(temp) : 0.0f;

    int eos_step = -1;
    int used = 0;

    for (int i = 0; i < max_frames; i++) {
        float normed[FLOWLM_D_MODEL];
        layernorm_forward(x, 1, FLOWLM_D_MODEL, fm->out_norm_w, fm->out_norm_b, 1e-5f, normed);
        if (i == 0 && out_first_cond) {
            memcpy(out_first_cond, normed, sizeof(normed));
        }

        float eos = 0.0f;
        for (int d = 0; d < FLOWLM_D_MODEL; d++) eos += fm->out_eos_w[d] * normed[d];
        eos += fm->out_eos_b ? fm->out_eos_b[0] : 0.0f;
        if (i == 0 && out_first_eos_logit) *out_first_eos_logit = eos;

        if (eos_enabled && i + 1 >= eos_min_frames && eos >= eos_threshold) {
            if (eos_step < 0) eos_step = i;
        }

        float latent[FLOWLM_LATENT_DIM];
        for (int d = 0; d < FLOWLM_LATENT_DIM; d += 2) {
            float z0 = 0.0f;
            float z1 = 0.0f;
            if (std > 0.0f) {
                float u1 = rng_next_f01(&rng);
                float u2 = rng_next_f01(&rng);
                float r = sqrtf(-2.0f * logf(u1));
                float theta = 2.0f * (float)M_PI * u2;
                z0 = r * cosf(theta) * std;
                z1 = r * sinf(theta) * std;
            }
            if (noise_clamp > 0.0f) {
                if (z0 < -noise_clamp) z0 = -noise_clamp;
                if (z0 > noise_clamp) z0 = noise_clamp;
                if (z1 < -noise_clamp) z1 = -noise_clamp;
                if (z1 > noise_clamp) z1 = noise_clamp;
            }
            latent[d] = z0;
            if (d + 1 < FLOWLM_LATENT_DIM) latent[d + 1] = z1;
        }

        float *flow_out = (i == 0) ? out_first_flow : NULL;
        lsd_decode(fm, normed, lsd_steps, latent, flow_out);
        memcpy(out_latents + (size_t)i * FLOWLM_LATENT_DIM, latent, sizeof(latent));

        used = i + 1;
        if (eos_step >= 0 && i >= eos_step + eos_after) break;

        linear_forward(fm->input_linear_w, NULL, FLOWLM_D_MODEL, FLOWLM_LATENT_DIM,
                       latent, 1, x);
        if (transformer_forward_step_cached(fm, cache, x) != 0) {
            kv_cache_free(cache);
            return -1;
        }
    }

    *out_frames_used = used;
    kv_cache_free(cache);
    return 0;
}

void ptts_flowlm_scale_latents(const ptts_flowlm *fm, const float *in_latents,
                               int frames, float *out_latents) {
    if (!fm || !in_latents || !out_latents || frames <= 0) return;
    for (int i = 0; i < frames; i++) {
        const float *src = in_latents + (size_t)i * FLOWLM_LATENT_DIM;
        float *dst = out_latents + (size_t)i * FLOWLM_LATENT_DIM;
        for (int d = 0; d < FLOWLM_LATENT_DIM; d++) {
            dst[d] = src[d] * fm->emb_std[d] + fm->emb_mean[d];
        }
    }
}
