#ifndef PTTS_SPM_H
#define PTTS_SPM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ptts_spm ptts_spm;

ptts_spm *ptts_spm_load(const char *path);
void ptts_spm_free(ptts_spm *spm);

/* Encode text into token IDs. Allocates *out_ids (caller frees). Returns 0 on success. */
int ptts_spm_encode(const ptts_spm *spm, const char *text, int **out_ids, int *out_len);

/* Access piece bytes for a given ID. Returns pointer owned by spm. */
const char *ptts_spm_piece(const ptts_spm *spm, int id, int *out_len);

int ptts_spm_vocab_size(const ptts_spm *spm);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_SPM_H */
