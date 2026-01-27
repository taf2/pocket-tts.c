#include "ptts.h"
#include "ptts_flowlm.h"
#include "ptts_internal.h"
#include "ptts_mimi.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Error handling
 * ======================================================================== */

static char g_error_msg[256] = {0};
static int g_timing_inited = 0;
static int g_timing_enabled = 0;

const char *ptts_get_error(void) {
    return g_error_msg;
}

int ptts_timing_enabled(void) {
    if (!g_timing_inited) {
        const char *v = getenv("PTTS_TIMING");
        g_timing_enabled = (v && v[0] && strcmp(v, "0") != 0);
        g_timing_inited = 1;
    }
    return g_timing_enabled;
}

double ptts_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void set_error(const char *msg) {
    strncpy(g_error_msg, msg, sizeof(g_error_msg) - 1);
    g_error_msg[sizeof(g_error_msg) - 1] = '\0';
}

/* ========================================================================
 * Helpers
 * ======================================================================== */

static int file_exists(const char *path) {
    struct stat st;
    return path && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int dir_exists(const char *path) {
    struct stat st;
    return path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int has_suffix(const char *s, const char *suffix) {
    size_t len = strlen(s);
    size_t slen = strlen(suffix);
    if (len < slen) return 0;
    return strcmp(s + len - slen, suffix) == 0;
}

static char *join_path(const char *a, const char *b) {
    size_t alen = strlen(a);
    size_t blen = strlen(b);
    size_t need = alen + blen + 2;
    char *out = (char *)malloc(need);
    if (!out) return NULL;
    snprintf(out, need, "%s/%s", a, b);
    return out;
}

static char *find_weights_file(const char *model_dir) {
    if (!model_dir) return NULL;

    /* Direct .safetensors path */
    if (has_suffix(model_dir, ".safetensors") && file_exists(model_dir)) {
        return strdup(model_dir);
    }

    if (!dir_exists(model_dir)) return NULL;

    /* Preferred filename */
    char *preferred = join_path(model_dir, "tts_b6369a24.safetensors");
    if (preferred && file_exists(preferred)) return preferred;
    free(preferred);

    /* First .safetensors file in directory */
    DIR *dir = opendir(model_dir);
    if (!dir) return NULL;

    struct dirent *ent;
    char *found = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!has_suffix(ent->d_name, ".safetensors")) continue;
        char *path = join_path(model_dir, ent->d_name);
        if (path && file_exists(path)) {
            found = path;
            break;
        }
        free(path);
    }

    closedir(dir);
    return found;
}

static char *dirname_from_path(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) return strdup(".");
    size_t len = (size_t)(slash - path);
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, path, len);
    out[len] = '\0';
    return out;
}

static char *find_tokenizer_file(const char *model_dir) {
    if (!model_dir) return NULL;

    char *base_dir = NULL;
    if (has_suffix(model_dir, ".safetensors")) {
        base_dir = dirname_from_path(model_dir);
    } else {
        base_dir = strdup(model_dir);
    }
    if (!base_dir) return NULL;

    char *candidate = join_path(base_dir, "tokenizer.model");
    free(base_dir);
    if (candidate && file_exists(candidate)) return candidate;
    free(candidate);
    return NULL;
}

static int voice_is_disabled(const char *voice) {
    if (!voice || voice[0] == '\0') return 0;
    return (strcmp(voice, "none") == 0 ||
            strcmp(voice, "off") == 0 ||
            strcmp(voice, "null") == 0);
}

static char *resolve_voice_path(const ptts_ctx *ctx, const char *voice_path) {
    const char *name = (voice_path && voice_path[0]) ? voice_path : "alba";
    if (voice_is_disabled(name)) return NULL;

    if (file_exists(name)) return strdup(name);

    if (!ctx || !ctx->model_dir) return NULL;

    char *base_dir = NULL;
    if (has_suffix(ctx->model_dir, ".safetensors")) {
        base_dir = dirname_from_path(ctx->model_dir);
    } else {
        base_dir = strdup(ctx->model_dir);
    }
    if (!base_dir) return NULL;

    if (strchr(name, '/')) {
        char *cand = join_path(base_dir, name);
        if (cand && file_exists(cand)) {
            free(base_dir);
            return cand;
        }
        free(cand);
    }

    if (has_suffix(name, ".safetensors")) {
        char *cand = join_path(base_dir, name);
        if (cand && file_exists(cand)) {
            free(base_dir);
            return cand;
        }
        free(cand);
    }

    size_t need = strlen(base_dir) + strlen(name) + 32;
    char *cand = (char *)malloc(need);
    if (!cand) {
        free(base_dir);
        return NULL;
    }
    snprintf(cand, need, "%s/embeddings/%s.safetensors", base_dir, name);
    if (file_exists(cand)) {
        free(base_dir);
        return cand;
    }
    snprintf(cand, need, "%s/voices/%s.safetensors", base_dir, name);
    if (file_exists(cand)) {
        free(base_dir);
        return cand;
    }
    snprintf(cand, need, "%s/%s.safetensors", base_dir, name);
    if (file_exists(cand)) {
        free(base_dir);
        return cand;
    }

    free(cand);
    free(base_dir);
    return NULL;
}

/* ========================================================================
 * Prompt preparation + heuristics
 * ======================================================================== */

char *ptts_prepare_text(const char *text, int *out_word_count, int *out_eos_after) {
    if (!text) return NULL;

    size_t len = strlen(text);
    char *buf = (char *)malloc(len + 32);
    if (!buf) return NULL;

    int out = 0;
    int in_space = 1;
    int words = 0;
    for (size_t i = 0; i < len; i++) {
        char c = text[i];
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (c == ' ') {
            if (!in_space) {
                buf[out++] = ' ';
                in_space = 1;
            }
            continue;
        }
        if (in_space) words++;
        in_space = 0;
        buf[out++] = c;
    }
    if (out > 0 && buf[out - 1] == ' ') out--;
    buf[out] = '\0';
    if (out == 0) {
        free(buf);
        set_error("Text prompt cannot be empty");
        return NULL;
    }

    for (int i = 0; i < out; i++) {
        unsigned char c = (unsigned char)buf[i];
        if (isalpha(c)) {
            buf[i] = (char)toupper(c);
            break;
        }
    }

    int last = out - 1;
    while (last >= 0 && buf[last] == ' ') last--;
    if (last >= 0 && isalnum((unsigned char)buf[last])) {
        buf[out++] = '.';
        buf[out] = '\0';
    }

    int eos_after = (words <= 4) ? 5 : 3;

    if (words < 5) {
        char *pref = (char *)malloc((size_t)out + 9);
        if (!pref) {
            free(buf);
            return NULL;
        }
        memset(pref, ' ', 8);
        memcpy(pref + 8, buf, (size_t)out + 1);
        free(buf);
        buf = pref;
    }

    if (out_word_count) *out_word_count = words;
    if (out_eos_after) *out_eos_after = eos_after;
    return buf;
}

int ptts_estimate_frames(int word_count) {
    if (word_count < 1) word_count = 1;
    float gen_len_sec = (float)word_count * 1.0f + 2.0f;
    int frames = (int)(gen_len_sec * 12.5f);
    if (frames < 1) frames = 1;
    return frames;
}

int ptts_load_voice_conditioning(ptts_ctx *ctx, const char *voice_path,
                                 float **out_cond, int *out_len) {
    if (!out_cond || !out_len) return -1;
    *out_cond = NULL;
    *out_len = 0;

    const char *name = (voice_path && voice_path[0]) ? voice_path : "alba";
    if (voice_is_disabled(name)) {
        return 0;
    }

    char *resolved = resolve_voice_path(ctx, name);
    if (!resolved) {
        set_error("Voice prompt not found (run ./download_model.sh --voice alba or pass --voice PATH)");
        return -1;
    }

    safetensors_file_t *sf = safetensors_open(resolved);
    if (!sf) {
        free(resolved);
        set_error("Failed to open voice prompt file");
        return -1;
    }

    const safetensor_t *t = safetensors_find(sf, "audio_prompt");
    if (!t) {
        safetensors_close(sf);
        free(resolved);
        set_error("Voice prompt missing audio_prompt tensor");
        return -1;
    }

    int64_t frames = 0;
    int64_t dim = 0;
    if (t->ndim == 3) {
        if (t->shape[0] != 1) {
            safetensors_close(sf);
            free(resolved);
            set_error("Voice prompt batch dimension must be 1");
            return -1;
        }
        frames = t->shape[1];
        dim = t->shape[2];
    } else if (t->ndim == 2) {
        frames = t->shape[0];
        dim = t->shape[1];
    } else {
        safetensors_close(sf);
        free(resolved);
        set_error("Voice prompt has unexpected rank");
        return -1;
    }

    if (dim != PTTS_FLOWLM_DIM) {
        safetensors_close(sf);
        free(resolved);
        set_error("Voice prompt has unexpected embedding dim");
        return -1;
    }

    float *prompt = safetensors_get_f32(sf, t);
    safetensors_close(sf);
    free(resolved);
    if (!prompt) {
        set_error("Failed to load voice prompt tensor");
        return -1;
    }

    *out_cond = prompt;
    *out_len = (int)frames;
    return 0;
}

/* ========================================================================
 * Core API
 * ======================================================================== */

ptts_ctx *ptts_load_dir(const char *model_dir) {
    if (!model_dir) {
        set_error("Model directory required");
        return NULL;
    }

    char *weights_path = find_weights_file(model_dir);
    if (!weights_path) {
        set_error("No .safetensors file found in model directory");
        return NULL;
    }

    safetensors_file_t *sf = safetensors_open(weights_path);
    if (!sf) {
        free(weights_path);
        set_error("Failed to open safetensors file");
        return NULL;
    }

    ptts_ctx *ctx = (ptts_ctx *)calloc(1, sizeof(ptts_ctx));
    if (!ctx) {
        safetensors_close(sf);
        free(weights_path);
        set_error("Out of memory");
        return NULL;
    }

    ctx->model_dir = strdup(model_dir);
    ctx->weights_path = weights_path;
    ctx->weights = sf;
    ctx->sample_rate = PTTS_DEFAULT_SAMPLE_RATE;

    ctx->tokenizer_path = find_tokenizer_file(model_dir);
    if (ctx->tokenizer_path) {
        ctx->tokenizer = ptts_spm_load(ctx->tokenizer_path);
        if (!ctx->tokenizer) {
            free(ctx->tokenizer_path);
            ctx->tokenizer_path = NULL;
        }
    }

    return ctx;
}

void ptts_free(ptts_ctx *ctx) {
    if (!ctx) return;
    safetensors_close(ctx->weights);
    free(ctx->weights_path);
    free(ctx->tokenizer_path);
    ptts_spm_free(ctx->tokenizer);
    free(ctx->model_dir);
    free(ctx);
}

int ptts_print_info(const ptts_ctx *ctx) {
    if (!ctx || !ctx->weights) return -1;
    printf("Pocket-TTS model info\n");
    printf("  Weights: %s\n", ctx->weights_path ? ctx->weights_path : "(none)");
    printf("  Tokenizer: %s\n", ctx->tokenizer_path ? ctx->tokenizer_path : "(not found)");
    if (ctx->tokenizer) {
        printf("  Vocab size: %d\n", ptts_spm_vocab_size(ctx->tokenizer));
    }
    printf("  Tensors: %d\n", ctx->weights->num_tensors);
    printf("  Sample rate (default): %d\n", ctx->sample_rate);
    return 0;
}

int ptts_list_tensors(const ptts_ctx *ctx) {
    if (!ctx || !ctx->weights) return -1;
    safetensors_print_all(ctx->weights);
    return 0;
}

int ptts_list_tensors_matching(const ptts_ctx *ctx, const char *substr) {
    if (!ctx || !ctx->weights || !substr) return -1;
    int count = 0;
    for (int i = 0; i < ctx->weights->num_tensors; i++) {
        const char *name = ctx->weights->tensors[i].name;
        if (strstr(name, substr)) {
            safetensor_print(&ctx->weights->tensors[i]);
            count++;
        }
    }
    return count;
}

/* ========================================================================
 * Weight verification (FlowLM + Mimi)
 * ======================================================================== */

static int ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t tlen = strlen(suffix);
    if (slen < tlen) return 0;
    return strcmp(s + slen - tlen, suffix) == 0;
}

static const safetensor_t *find_tensor_exact(const ptts_ctx *ctx, const char *name) {
    return safetensors_find(ctx->weights, name);
}

static const safetensor_t *find_tensor_with_prefix(const ptts_ctx *ctx, const char *prefix,
                                                   const char *name, const char **found_name) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s%s", prefix, name);
    const safetensor_t *t = safetensors_find(ctx->weights, buf);
    if (t && found_name) *found_name = t->name;
    return t;
}

static const safetensor_t *find_tensor_suffix(const ptts_ctx *ctx, const char *suffix,
                                              const char **found_name, int *ambiguous) {
    const safetensor_t *match = NULL;
    for (int i = 0; i < ctx->weights->num_tensors; i++) {
        const safetensor_t *t = &ctx->weights->tensors[i];
        if (ends_with(t->name, suffix)) {
            if (match) {
                if (ambiguous) *ambiguous = 1;
                return NULL;
            }
            match = t;
        }
    }
    if (match && found_name) *found_name = match->name;
    return match;
}

static const safetensor_t *find_tensor_flowlm(const ptts_ctx *ctx, const char *name,
                                              const char **found_name, int *ambiguous) {
    const safetensor_t *t = find_tensor_exact(ctx, name);
    if (t) { if (found_name) *found_name = t->name; return t; }
    t = find_tensor_with_prefix(ctx, "flow_lm.", name, found_name);
    if (t) return t;
    t = find_tensor_suffix(ctx, name, found_name, ambiguous);
    return t;
}

static const safetensor_t *find_tensor_mimi(const ptts_ctx *ctx, const char *name,
                                            const char **found_name, int *ambiguous) {
    const safetensor_t *t = find_tensor_exact(ctx, name);
    if (t) { if (found_name) *found_name = t->name; return t; }
    t = find_tensor_with_prefix(ctx, "mimi.", name, found_name);
    if (t) return t;
    t = find_tensor_with_prefix(ctx, "model.", name, found_name);
    if (t) return t;
    t = find_tensor_suffix(ctx, name, found_name, ambiguous);
    return t;
}

static int shape_matches(const safetensor_t *t, int ndim, const int64_t *shape) {
    if (!t || t->ndim != ndim) return 0;
    for (int i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) return 0;
    }
    return 1;
}

static void print_shape_expected(int ndim, const int64_t *shape) {
    fprintf(stderr, "[");
    for (int i = 0; i < ndim; i++) {
        fprintf(stderr, "%lld", (long long)shape[i]);
        if (i < ndim - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]");
}

static void print_shape_actual(const safetensor_t *t) {
    if (!t) return;
    fprintf(stderr, "[");
    for (int i = 0; i < t->ndim; i++) {
        fprintf(stderr, "%lld", (long long)t->shape[i]);
        if (i < t->ndim - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]");
}

static int check_tensor(const ptts_ctx *ctx,
                        const char *name,
                        int ndim,
                        const int64_t *shape,
                        const safetensor_t *(*finder)(const ptts_ctx *, const char *, const char **, int *),
                        int verbose,
                        int *missing,
                        int *mismatch,
                        int *ambiguous) {
    const char *found = NULL;
    int amb = 0;
    const safetensor_t *t = finder(ctx, name, &found, &amb);
    if (amb) {
        if (verbose) {
            fprintf(stderr, "Ambiguous tensor match for %s\n", name);
        }
        if (ambiguous) (*ambiguous)++;
        return -1;
    }
    if (!t) {
        if (verbose) {
            fprintf(stderr, "Missing tensor: %s\n", name);
        }
        if (missing) (*missing)++;
        return -1;
    }
    if (!shape_matches(t, ndim, shape)) {
        if (verbose) {
            fprintf(stderr, "Shape mismatch for %s (%s): expected ", name, found ? found : "?");
            print_shape_expected(ndim, shape);
            fprintf(stderr, ", got ");
            print_shape_actual(t);
            fprintf(stderr, "\n");
        }
        if (mismatch) (*mismatch)++;
        return -1;
    }
    return 0;
}

static int verify_flowlm(const ptts_ctx *ctx, int verbose) {
    const int64_t text_dim = 1024;
    const int64_t text_vocab = 4000;
    const int64_t latent_dim = 32;
    const int64_t flow_dim = 512;
    const int64_t flow_depth = 6;
    const int64_t d_model = 1024;
    const int64_t num_layers = 6;
    const int64_t hidden_scale = 4;

    int missing = 0, mismatch = 0, ambiguous = 0;

    int64_t shape1[1];
    int64_t shape2[2];

    /* Conditioner + speaker */
    shape2[0] = text_vocab + 1;
    shape2[1] = text_dim;
    check_tensor(ctx, "conditioner.embed.weight", 2, shape2, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);

    shape2[0] = text_dim;
    shape2[1] = 512;
    check_tensor(ctx, "speaker_proj_weight", 2, shape2, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);

    /* Flow net */
    shape2[0] = flow_dim;
    shape2[1] = d_model;
    check_tensor(ctx, "flow_net.cond_embed.weight", 2, shape2, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);
    shape1[0] = flow_dim;
    check_tensor(ctx, "flow_net.cond_embed.bias", 1, shape1, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);

    shape2[0] = flow_dim;
    shape2[1] = latent_dim;
    check_tensor(ctx, "flow_net.input_proj.weight", 2, shape2, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);
    shape1[0] = flow_dim;
    check_tensor(ctx, "flow_net.input_proj.bias", 1, shape1, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);

    for (int t = 0; t < 2; t++) {
        char name[128];
        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.0.weight", t);
        shape2[0] = flow_dim;
        shape2[1] = 256;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.0.bias", t);
        shape1[0] = flow_dim;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.2.weight", t);
        shape2[0] = flow_dim;
        shape2[1] = flow_dim;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.2.bias", t);
        shape1[0] = flow_dim;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.time_embed.%d.mlp.3.alpha", t);
        shape1[0] = flow_dim;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);
    }

    for (int i = 0; i < flow_depth; i++) {
        char name[160];
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.in_ln.weight", i);
        shape1[0] = flow_dim;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.in_ln.bias", i);
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.0.weight", i);
        shape2[0] = flow_dim;
        shape2[1] = flow_dim;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.0.bias", i);
        shape1[0] = flow_dim;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.2.weight", i);
        shape2[0] = flow_dim;
        shape2[1] = flow_dim;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.mlp.2.bias", i);
        shape1[0] = flow_dim;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.adaLN_modulation.1.weight", i);
        shape2[0] = flow_dim * 3;
        shape2[1] = flow_dim;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);
        snprintf(name, sizeof(name), "flow_net.res_blocks.%d.adaLN_modulation.1.bias", i);
        shape1[0] = flow_dim * 3;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm,
                     verbose, &missing, &mismatch, &ambiguous);
    }

    shape2[0] = latent_dim;
    shape2[1] = flow_dim;
    check_tensor(ctx, "flow_net.final_layer.linear.weight", 2, shape2, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);
    shape1[0] = latent_dim;
    check_tensor(ctx, "flow_net.final_layer.linear.bias", 1, shape1, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);

    shape2[0] = flow_dim * 2;
    shape2[1] = flow_dim;
    check_tensor(ctx, "flow_net.final_layer.adaLN_modulation.1.weight", 2, shape2, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);
    shape1[0] = flow_dim * 2;
    check_tensor(ctx, "flow_net.final_layer.adaLN_modulation.1.bias", 1, shape1, find_tensor_flowlm,
                 verbose, &missing, &mismatch, &ambiguous);

    /* FlowLM core params */
    shape1[0] = latent_dim;
    check_tensor(ctx, "emb_std", 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
    check_tensor(ctx, "emb_mean", 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
    check_tensor(ctx, "bos_emb", 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

    shape2[0] = d_model;
    shape2[1] = latent_dim;
    check_tensor(ctx, "input_linear.weight", 2, shape2, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

    shape1[0] = d_model;
    check_tensor(ctx, "out_norm.weight", 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
    check_tensor(ctx, "out_norm.bias", 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

    shape2[0] = 1;
    shape2[1] = d_model;
    check_tensor(ctx, "out_eos.weight", 2, shape2, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
    shape1[0] = 1;
    check_tensor(ctx, "out_eos.bias", 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

    for (int i = 0; i < num_layers; i++) {
        char name[160];
        snprintf(name, sizeof(name), "transformer.layers.%d.self_attn.in_proj.weight", i);
        shape2[0] = d_model * 3;
        shape2[1] = d_model;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "transformer.layers.%d.self_attn.out_proj.weight", i);
        shape2[0] = d_model;
        shape2[1] = d_model;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "transformer.layers.%d.norm1.weight", i);
        shape1[0] = d_model;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
        snprintf(name, sizeof(name), "transformer.layers.%d.norm1.bias", i);
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "transformer.layers.%d.norm2.weight", i);
        shape1[0] = d_model;
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
        snprintf(name, sizeof(name), "transformer.layers.%d.norm2.bias", i);
        check_tensor(ctx, name, 1, shape1, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "transformer.layers.%d.linear1.weight", i);
        shape2[0] = d_model * hidden_scale;
        shape2[1] = d_model;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);

        snprintf(name, sizeof(name), "transformer.layers.%d.linear2.weight", i);
        shape2[0] = d_model;
        shape2[1] = d_model * hidden_scale;
        check_tensor(ctx, name, 2, shape2, find_tensor_flowlm, verbose, &missing, &mismatch, &ambiguous);
    }

    if (verbose) {
        fprintf(stderr, "FlowLM verify: missing=%d mismatch=%d ambiguous=%d\n",
                missing, mismatch, ambiguous);
    }
    return missing + mismatch + ambiguous;
}

static void expect_conv1d(const ptts_ctx *ctx, const char *base, int out_ch, int in_ch, int k,
                          int bias, int verbose, int *missing, int *mismatch, int *ambiguous,
                          const safetensor_t *(*finder)(const ptts_ctx *, const char *, const char **, int *)) {
    char name[256];
    int64_t shape2[2];
    int64_t shape3[3];

    snprintf(name, sizeof(name), "%s.conv.weight", base);
    shape3[0] = out_ch;
    shape3[1] = in_ch;
    shape3[2] = k;
    check_tensor(ctx, name, 3, shape3, finder, verbose, missing, mismatch, ambiguous);

    if (bias) {
        snprintf(name, sizeof(name), "%s.conv.bias", base);
        shape2[0] = out_ch;
        check_tensor(ctx, name, 1, shape2, finder, verbose, missing, mismatch, ambiguous);
    }
}

static void expect_convtr1d(const ptts_ctx *ctx, const char *base, int in_ch, int out_ch, int k,
                            int bias, int verbose, int *missing, int *mismatch, int *ambiguous,
                            const safetensor_t *(*finder)(const ptts_ctx *, const char *, const char **, int *)) {
    char name[256];
    int64_t shape3[3];
    int64_t shape1[1];
    snprintf(name, sizeof(name), "%s.convtr.weight", base);
    shape3[0] = in_ch;
    shape3[1] = out_ch;
    shape3[2] = k;
    check_tensor(ctx, name, 3, shape3, finder, verbose, missing, mismatch, ambiguous);
    if (bias) {
        snprintf(name, sizeof(name), "%s.convtr.bias", base);
        shape1[0] = out_ch;
        check_tensor(ctx, name, 1, shape1, finder, verbose, missing, mismatch, ambiguous);
    }
}

static void expect_resblock(const ptts_ctx *ctx, const char *base, int dim, int compress,
                            int res_kernel, int verbose, int *missing, int *mismatch, int *ambiguous,
                            const safetensor_t *(*finder)(const ptts_ctx *, const char *, const char **, int *)) {
    int hidden = dim / compress;
    char name[256];
    int64_t shape3[3];
    int64_t shape1[1];

    snprintf(name, sizeof(name), "%s.block.1.conv.weight", base);
    shape3[0] = hidden;
    shape3[1] = dim;
    shape3[2] = res_kernel;
    check_tensor(ctx, name, 3, shape3, finder, verbose, missing, mismatch, ambiguous);

    snprintf(name, sizeof(name), "%s.block.1.conv.bias", base);
    shape1[0] = hidden;
    check_tensor(ctx, name, 1, shape1, finder, verbose, missing, mismatch, ambiguous);

    snprintf(name, sizeof(name), "%s.block.3.conv.weight", base);
    shape3[0] = dim;
    shape3[1] = hidden;
    shape3[2] = 1;
    check_tensor(ctx, name, 3, shape3, finder, verbose, missing, mismatch, ambiguous);

    snprintf(name, sizeof(name), "%s.block.3.conv.bias", base);
    shape1[0] = dim;
    check_tensor(ctx, name, 1, shape1, finder, verbose, missing, mismatch, ambiguous);
}

static int verify_mimi_transformer(const ptts_ctx *ctx, const char *prefix,
                                   int d_model, int num_layers, int hidden_ff, int layer_scale_on,
                                   int verbose, int *missing, int *mismatch, int *ambiguous) {
    char name[256];
    int64_t shape1[1];
    int64_t shape2[2];

    for (int i = 0; i < num_layers; i++) {
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.self_attn.in_proj.weight", prefix, i);
        shape2[0] = d_model * 3;
        shape2[1] = d_model;
        check_tensor(ctx, name, 2, shape2, find_tensor_mimi, verbose, missing, mismatch, ambiguous);

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.self_attn.out_proj.weight", prefix, i);
        shape2[0] = d_model;
        shape2[1] = d_model;
        check_tensor(ctx, name, 2, shape2, find_tensor_mimi, verbose, missing, mismatch, ambiguous);

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.norm1.weight", prefix, i);
        shape1[0] = d_model;
        check_tensor(ctx, name, 1, shape1, find_tensor_mimi, verbose, missing, mismatch, ambiguous);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.norm1.bias", prefix, i);
        check_tensor(ctx, name, 1, shape1, find_tensor_mimi, verbose, missing, mismatch, ambiguous);

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.norm2.weight", prefix, i);
        shape1[0] = d_model;
        check_tensor(ctx, name, 1, shape1, find_tensor_mimi, verbose, missing, mismatch, ambiguous);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.norm2.bias", prefix, i);
        check_tensor(ctx, name, 1, shape1, find_tensor_mimi, verbose, missing, mismatch, ambiguous);

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.linear1.weight", prefix, i);
        shape2[0] = hidden_ff;
        shape2[1] = d_model;
        check_tensor(ctx, name, 2, shape2, find_tensor_mimi, verbose, missing, mismatch, ambiguous);

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.linear2.weight", prefix, i);
        shape2[0] = d_model;
        shape2[1] = hidden_ff;
        check_tensor(ctx, name, 2, shape2, find_tensor_mimi, verbose, missing, mismatch, ambiguous);

        if (layer_scale_on) {
            snprintf(name, sizeof(name), "%s.transformer.layers.%d.layer_scale_1.scale", prefix, i);
            shape1[0] = d_model;
            check_tensor(ctx, name, 1, shape1, find_tensor_mimi, verbose, missing, mismatch, ambiguous);
            snprintf(name, sizeof(name), "%s.transformer.layers.%d.layer_scale_2.scale", prefix, i);
            check_tensor(ctx, name, 1, shape1, find_tensor_mimi, verbose, missing, mismatch, ambiguous);
        }
    }
    return 0;
}

static int verify_mimi(const ptts_ctx *ctx, int verbose) {
    const int channels = 1;
    const int dimension = 512;
    const int n_filters = 64;
    const int n_residual_layers = 1;
    const int ratios[3] = {6, 5, 4};
    const int kernel_size = 7;
    const int last_kernel_size = 3;
    const int residual_kernel = 3;
    const int compress = 2;
    const int d_model = 512;
    const int num_layers = 2;
    const int hidden_ff = 2048;
    const int layer_scale_on = 1;

    int missing = 0, mismatch = 0, ambiguous = 0;

    /* Downsample/Upsample (encoder_frame_rate 200 -> frame_rate 12.5 => stride 16) */
    expect_conv1d(ctx, "downsample.conv", dimension, dimension, 32, 0,
                  verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
    expect_convtr1d(ctx, "upsample.convtr", dimension, 1, 32, 0,
                    verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);

    /* Encoder */
    int idx = 0;
    char base[128];
    expect_conv1d(ctx, "encoder.model.0", n_filters, channels, kernel_size, 1,
                  verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
    idx = 1;
    int mult = 1;
    for (int r = 2; r >= 0; r--) {
        int ratio = ratios[r];
        for (int j = 0; j < n_residual_layers; j++) {
            snprintf(base, sizeof(base), "encoder.model.%d", idx);
            expect_resblock(ctx, base, mult * n_filters, compress, residual_kernel,
                            verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
            idx++;
        }
        idx++; /* ELU */
        snprintf(base, sizeof(base), "encoder.model.%d", idx);
        expect_conv1d(ctx, base, mult * n_filters * 2, mult * n_filters, ratio * 2, 1,
                      verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
        idx++;
        mult *= 2;
    }
    idx++; /* ELU */
    snprintf(base, sizeof(base), "encoder.model.%d", idx);
    expect_conv1d(ctx, base, dimension, mult * n_filters, last_kernel_size, 1,
                  verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);

    /* Decoder */
    idx = 0;
    mult = 1 << 3; /* 2^len(ratios) */
    expect_conv1d(ctx, "decoder.model.0", mult * n_filters, dimension, kernel_size, 1,
                  verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
    idx = 1;
    for (int r = 0; r < 3; r++) {
        int ratio = ratios[r];
        idx++; /* ELU */
        snprintf(base, sizeof(base), "decoder.model.%d", idx);
        expect_convtr1d(ctx, base, mult * n_filters, mult * n_filters / 2, ratio * 2, 1,
                        verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
        idx++;
        for (int j = 0; j < n_residual_layers; j++) {
            snprintf(base, sizeof(base), "decoder.model.%d", idx);
            expect_resblock(ctx, base, mult * n_filters / 2, compress, residual_kernel,
                            verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);
            idx++;
        }
        mult /= 2;
    }
    idx++; /* ELU */
    snprintf(base, sizeof(base), "decoder.model.%d", idx);
    expect_conv1d(ctx, base, channels, n_filters, last_kernel_size, 1,
                  verbose, &missing, &mismatch, &ambiguous, find_tensor_mimi);

    /* Mimi transformers */
    verify_mimi_transformer(ctx, "encoder_transformer", d_model, num_layers, hidden_ff, layer_scale_on,
                            verbose, &missing, &mismatch, &ambiguous);
    verify_mimi_transformer(ctx, "decoder_transformer", d_model, num_layers, hidden_ff, layer_scale_on,
                            verbose, &missing, &mismatch, &ambiguous);

    if (verbose) {
        fprintf(stderr, "Mimi verify: missing=%d mismatch=%d ambiguous=%d\n",
                missing, mismatch, ambiguous);
    }
    return missing + mismatch + ambiguous;
}

int ptts_verify_weights(const ptts_ctx *ctx, int verbose) {
    if (!ctx || !ctx->weights) return -1;
    int errs = 0;
    errs += verify_flowlm(ctx, verbose);
    errs += verify_mimi(ctx, verbose);
    return errs == 0 ? 0 : -1;
}

int ptts_tokenize(ptts_ctx *ctx, const char *text, int **out_ids, int *out_len) {
    if (!ctx || !text || !out_ids || !out_len) return -1;
    if (!ctx->tokenizer) {
        set_error("Tokenizer not loaded (tokenizer.model missing or failed to parse)");
        return -1;
    }
    if (ptts_spm_encode(ctx->tokenizer, text, out_ids, out_len) != 0) {
        set_error("Tokenization failed");
        return -1;
    }
    return 0;
}

const char *ptts_token_piece(ptts_ctx *ctx, int id, int *out_len) {
    if (!ctx || !ctx->tokenizer) return NULL;
    return ptts_spm_piece(ctx->tokenizer, id, out_len);
}

ptts_audio *ptts_generate(ptts_ctx *ctx, const char *text,
                          const char *voice_path, const ptts_params *params) {
    if (!ctx || !text) {
        set_error("Text required");
        return NULL;
    }

    ptts_params p = PTTS_PARAMS_DEFAULT;
    if (params) p = *params;
    if (p.num_frames < 0) p.num_frames = 0;
    if (p.num_steps < 1) p.num_steps = 1;
    if (p.eos_min_frames < 1) p.eos_min_frames = 1;
    if (p.eos_after < 0) p.eos_after = 0;
    if (p.sample_rate <= 0) p.sample_rate = PTTS_DEFAULT_SAMPLE_RATE;
    if (p.temp < 0.0f) p.temp = 1.0f;

    int word_count = 0;
    int eos_after_guess = 0;
    char *prepared = ptts_prepare_text(text, &word_count, &eos_after_guess);
    if (!prepared) {
        return NULL;
    }

    int *ids = NULL;
    int n = 0;
    if (ptts_tokenize(ctx, prepared, &ids, &n) != 0) {
        free(prepared);
        return NULL;
    }
    free(prepared);

    if (p.num_frames <= 0) {
        p.num_frames = ptts_estimate_frames(word_count);
    }
    if (p.eos_after <= 0) p.eos_after = eos_after_guess;

    ptts_flowlm *fm = ptts_flowlm_load(ctx);
    if (!fm) {
        free(ids);
        set_error("Failed to load FlowLM weights");
        return NULL;
    }
    ptts_mimi *mm = ptts_mimi_load(ctx);
    if (!mm) {
        ptts_flowlm_free(fm);
        free(ids);
        set_error("Failed to load Mimi weights");
        return NULL;
    }

    float *voice_cond = NULL;
    int voice_len = 0;
    if (ptts_load_voice_conditioning(ctx, voice_path, &voice_cond, &voice_len) != 0) {
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        return NULL;
    }

    float *latents = (float *)malloc(sizeof(float) * 32 * (size_t)p.num_frames);
    if (!latents) {
        free(voice_cond);
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        set_error("Out of memory");
        return NULL;
    }

    int used_frames = 0;
    double t_start = 0.0;
    if (ptts_timing_enabled()) t_start = ptts_time_ms();
    if (ptts_flowlm_generate_latents(fm, ids, n, voice_cond, voice_len,
                                     p.num_frames, p.num_steps, p.temp, p.noise_clamp,
                                     p.seed, p.eos_enabled, p.eos_threshold, p.eos_min_frames,
                                     p.eos_after, latents, &used_frames, NULL, NULL, NULL) != 0) {
        free(latents);
        free(voice_cond);
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        set_error("FlowLM forward failed");
        return NULL;
    }
    if (ptts_timing_enabled()) {
        double t_end = ptts_time_ms();
        fprintf(stderr, "[ptts] FlowLM latents: %.2f ms (%d frames)\n",
                t_end - t_start, used_frames);
    }

    float *scaled = (float *)malloc(sizeof(float) * 32 * (size_t)used_frames);
    if (!scaled) {
        free(latents);
        free(voice_cond);
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        set_error("Out of memory");
        return NULL;
    }
    ptts_flowlm_scale_latents(fm, latents, used_frames, scaled);

    const int frame_samples = 16 * 6 * 5 * 4;
    int total_samples = frame_samples * used_frames;
    ptts_audio *audio = ptts_audio_create(p.sample_rate, 1, total_samples);
    if (!audio) {
        free(latents);
        free(voice_cond);
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        set_error("Out of memory");
        return NULL;
    }

    int wav_len = 0;
    if (ptts_timing_enabled()) t_start = ptts_time_ms();
    if (ptts_mimi_decode(mm, scaled, used_frames, audio->samples, &wav_len) != 0) {
        ptts_audio_free(audio);
        free(scaled);
        free(latents);
        free(voice_cond);
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        set_error("Mimi decode failed");
        return NULL;
    }
    if (ptts_timing_enabled()) {
        double t_end = ptts_time_ms();
        fprintf(stderr, "[ptts] Mimi decode: %.2f ms\n", t_end - t_start);
    }
    if (wav_len != total_samples) {
        ptts_audio_free(audio);
        free(scaled);
        free(latents);
        free(voice_cond);
        ptts_mimi_free(mm);
        ptts_flowlm_free(fm);
        free(ids);
        set_error("Unexpected Mimi output length");
        return NULL;
    }
    free(scaled);
    free(latents);
    free(voice_cond);
    ptts_mimi_free(mm);
    ptts_flowlm_free(fm);
    free(ids);
    return audio;
}

/* ========================================================================
 * Dummy generator (placeholder audio)
 * ======================================================================== */

static float char_frequency(unsigned char c) {
    if (c == ' ' || c == '\n' || c == '\t') return 0.0f;
    int bucket = (int)(c % 48);
    return 180.0f + (float)bucket * 12.0f;
}

ptts_audio *ptts_generate_dummy(const char *text, const ptts_params *params) {
    if (!text) {
        set_error("Text required");
        return NULL;
    }

    ptts_params p = PTTS_PARAMS_DEFAULT;
    if (params) p = *params;
    if (p.sample_rate <= 0) p.sample_rate = PTTS_DEFAULT_SAMPLE_RATE;

    const float char_sec = 0.06f;
    const float space_sec = 0.04f;
    const float tail_sec = 0.15f;

    size_t len = strlen(text);
    size_t total_samples = (size_t)(tail_sec * p.sample_rate);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        total_samples += (size_t)((c == ' ' || c == '\n' || c == '\t') ?
                                  (space_sec * p.sample_rate) : (char_sec * p.sample_rate));
    }

    ptts_audio *audio = ptts_audio_create(p.sample_rate, 1, (int)total_samples);
    if (!audio) {
        set_error("Out of memory");
        return NULL;
    }

    const int fade_samples = (int)(0.004f * p.sample_rate);
    const float amp = 0.2f;

    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        float freq = char_frequency(c);
        int seg = (int)(((c == ' ' || c == '\n' || c == '\t') ? space_sec : char_sec) * p.sample_rate);
        if (seg <= 0) continue;

        float phase = 0.0f;
        float phase_inc = (freq > 0.0f) ? (2.0f * (float)M_PI * freq / (float)p.sample_rate) : 0.0f;

        for (int s = 0; s < seg && pos < total_samples; s++, pos++) {
            float env = 1.0f;
            if (s < fade_samples) env = (float)s / (float)fade_samples;
            else if (s > seg - fade_samples) env = (float)(seg - s) / (float)fade_samples;
            if (env < 0.0f) env = 0.0f;

            float sample = 0.0f;
            if (freq > 0.0f) {
                sample = sinf(phase) * amp * env;
                phase += phase_inc;
            }
            audio->samples[pos] = sample;
        }
    }

    /* tail silence already zero-initialized */
    return audio;
}
