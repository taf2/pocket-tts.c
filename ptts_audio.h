#ifndef PTTS_AUDIO_H
#define PTTS_AUDIO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int sample_rate;
    int channels;
    int num_samples; /* per channel */
    float *samples;  /* interleaved, length = num_samples * channels */
} ptts_audio;

ptts_audio *ptts_audio_create(int sample_rate, int channels, int num_samples);
void ptts_audio_free(ptts_audio *audio);

/* Save audio as 16-bit PCM WAV. Returns 0 on success, -1 on error. */
int ptts_audio_save_wav(const ptts_audio *audio, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_AUDIO_H */
