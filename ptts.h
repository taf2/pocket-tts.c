#ifndef PTTS_H
#define PTTS_H

#include <stdint.h>
#include "ptts_audio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define PTTS_DEFAULT_SAMPLE_RATE 24000

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct ptts_ctx ptts_ctx;

/* ========================================================================
 * Parameters
 * ======================================================================== */

typedef struct {
    int sample_rate;  /* Output sample rate (default: 24000) */
    int num_steps;    /* Flow matching steps (placeholder) */
    int num_frames;   /* Number of frames to generate (80ms each) */
    int64_t seed;     /* Random seed (-1 for random) */
    float temp;       /* FlowLM noise temperature */
    float noise_clamp;/* Clamp noise to [-F, F] (0 disables) */
    int eos_enabled;  /* Enable EOS early stopping */
    float eos_threshold; /* Stop when eos_logit >= threshold */
    int eos_min_frames;  /* Minimum frames before EOS stop */
    int eos_after;    /* Frames to keep after EOS (0 = auto) */
} ptts_params;

#define PTTS_PARAMS_DEFAULT { PTTS_DEFAULT_SAMPLE_RATE, 1, 0, -1, 0.7f, 0.0f, 1, -4.0f, 1, 0 }

/* ========================================================================
 * Core API
 * ======================================================================== */

ptts_ctx *ptts_load_dir(const char *model_dir);
void ptts_free(ptts_ctx *ctx);

const char *ptts_get_error(void);

/* Inspect model */
int ptts_print_info(const ptts_ctx *ctx);
int ptts_list_tensors(const ptts_ctx *ctx);
int ptts_list_tensors_matching(const ptts_ctx *ctx, const char *substr);

/* Verify weights shapes against expected config (returns 0 on success). */
int ptts_verify_weights(const ptts_ctx *ctx, int verbose);

/* Tokenization (SentencePiece) */
int ptts_tokenize(ptts_ctx *ctx, const char *text, int **out_ids, int *out_len);
const char *ptts_token_piece(ptts_ctx *ctx, int id, int *out_len);

/* Prompt preparation + heuristics */
char *ptts_prepare_text(const char *text, int *out_word_count, int *out_eos_after);
int ptts_estimate_frames(int word_count);

/* Load voice conditioning (audio_prompt) from a safetensors file.
 * Returns 0 on success, fills out_cond with float32 buffer and out_len (frames).
 * Caller must free(*out_cond). voice_path may be NULL to use the default voice.
 * Use voice_path = "none" to disable voice conditioning. */
int ptts_load_voice_conditioning(ptts_ctx *ctx, const char *voice_path,
                                 float **out_cond, int *out_len);

/* Generate audio (WIP) */
ptts_audio *ptts_generate(ptts_ctx *ctx, const char *text,
                          const char *voice_path, const ptts_params *params);

/* Placeholder generator for pipeline testing */
ptts_audio *ptts_generate_dummy(const char *text, const ptts_params *params);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_H */
