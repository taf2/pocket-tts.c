#ifndef PTTS_FLOWLM_H
#define PTTS_FLOWLM_H

#include <stdint.h>
#include "ptts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ptts_flowlm ptts_flowlm;

#define PTTS_FLOWLM_DIM 1024
#define PTTS_FLOWLM_LATENT_DIM 32

ptts_flowlm *ptts_flowlm_load(ptts_ctx *ctx);
void ptts_flowlm_free(ptts_flowlm *fm);

/*
 * Run a single FlowLM step (non-streaming). Returns 0 on success.
 * tokens: input text tokens
 * token_len: number of tokens
 * lsd_steps: number of LSD decode steps (e.g., 4)
 * out_latent: output raw latent vector (length 32, before emb_std/emb_mean)
 * out_eos_logit: optional pointer for EOS logit
 */
int ptts_flowlm_forward_one(ptts_flowlm *fm, const int *tokens, int token_len,
                            const float *cond_prefix, int cond_len,
                            int lsd_steps, float temp, float noise_clamp,
                            int64_t seed, float *out_latent, float *out_eos_logit);

/* Generate next latent given previous latents (naive, non-streaming). */
int ptts_flowlm_forward_next(ptts_flowlm *fm, const int *tokens, int token_len,
                             const float *cond_prefix, int cond_len,
                             const float *prev_latents, int prev_len,
                             int lsd_steps, float temp, float noise_clamp,
                             int64_t *seed_io, float *out_latent, float *out_eos_logit);

/* Generate multiple latents with an internal KV cache (non-streaming). */
int ptts_flowlm_generate_latents(ptts_flowlm *fm, const int *tokens, int token_len,
                                 const float *cond_prefix, int cond_len,
                                 int max_frames, int lsd_steps, float temp, float noise_clamp,
                                 int64_t seed, int eos_enabled, float eos_threshold,
                                 int eos_min_frames, int eos_after,
                                 float *out_latents, int *out_frames_used,
                                 float *out_first_eos_logit,
                                 float *out_first_cond,
                                 float *out_first_flow);

/* Scale FlowLM latents to Mimi latent space. */
void ptts_flowlm_scale_latents(const ptts_flowlm *fm, const float *in_latents,
                               int frames, float *out_latents);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_FLOWLM_H */
