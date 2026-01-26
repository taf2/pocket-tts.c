#ifndef PTTS_INTERNAL_H
#define PTTS_INTERNAL_H

#include "ptts_safetensors.h"
#include "ptts_spm.h"

struct ptts_ctx {
    char *model_dir;
    char *weights_path;
    safetensors_file_t *weights;
    char *tokenizer_path;
    ptts_spm *tokenizer;
    int sample_rate;
};

#endif /* PTTS_INTERNAL_H */
