/*
 * Pocket-TTS CLI (WIP)
 *
 * Usage:
 *   ptts -d model_dir -p "text" -o out.wav [options]
 */

#include "ptts.h"
#include "ptts_flowlm.h"
#include "ptts_mimi.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OUTPUT_QUIET = 0,
    OUTPUT_NORMAL = 1,
    OUTPUT_VERBOSE = 2
} output_level_t;

static output_level_t output_level = OUTPUT_NORMAL;

static void print_usage(const char *prog) {
    printf("Pocket-TTS Pure C (WIP)\n");
    printf("Usage: %s -d model_dir -p \"text\" -o out.wav [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -d, --dir PATH        Model directory or .safetensors file\n");
    printf("  -p, --prompt TEXT     Text to synthesize\n");
    printf("  -o, --output PATH     Output WAV path\n");
    printf("      --voice NAME      Voice embedding name or .safetensors path (default: alba)\n");
    printf("      --info            Print model info\n");
    printf("      --list            List tensors in weights file\n");
    printf("      --find TEXT       List tensors whose names contain TEXT\n");
    printf("      --verify          Verify weights against expected shapes\n");
    printf("      --tokens          Print token IDs for the prompt\n");
    printf("      --flow-test       Run a single FlowLM step and print latent stats\n");
    printf("      --mimi-test       Run FlowLM + Mimi decoder transformer stats\n");
    printf("      --mimi-wave PATH  Write Mimi decode WAV to PATH (frames * 80ms)\n");
    printf("      --frames N        Number of FlowLM/Mimi frames (default: auto)\n");
    printf("      --cond-out PATH   Write first FlowLM condition vector (1024 floats)\n");
    printf("      --flow-out PATH   Write first FlowLM flow vector (32 floats)\n");
    printf("      --dummy           Generate placeholder audio (no model)\n");
    printf("  -r, --rate N          Sample rate for dummy generator (default: 24000)\n");
    printf("  -s, --steps N         Flow matching steps (placeholder)\n");
    printf("  -S, --seed N          Random seed (-1 for random)\n");
    printf("  -t, --temp F          Noise temperature for FlowLM (default: 1.0)\n");
    printf("      --noise-clamp F   Clamp noise to [-F, F] (default: 0, off)\n");
    printf("      --eos-threshold F Stop early if eos_logit >= F (default: -4.0)\n");
    printf("      --eos-min-frames N Minimum frames before EOS stop (default: 1)\n");
    printf("      --eos-after N    Frames to keep after EOS (default: auto)\n");
    printf("  -q, --quiet           Less output\n");
    printf("  -v, --verbose         More output\n");
    printf("  -h, --help            Show help\n");
}

#define LOG_NORMAL(...) do { if (output_level >= OUTPUT_NORMAL) fprintf(stderr, __VA_ARGS__); } while(0)
#define LOG_VERBOSE(...) do { if (output_level >= OUTPUT_VERBOSE) fprintf(stderr, __VA_ARGS__); } while(0)

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = NULL;
    const char *voice = NULL;
    int list_tensors = 0;
    int info_only = 0;
    int show_tokens = 0;
    int use_dummy = 0;
    int verify_weights = 0;
    int flow_test = 0;
    int mimi_test = 0;
    const char *mimi_wave = NULL;
    const char *find_pat = NULL;
    const char *latent_out = NULL;
    const char *cond_out = NULL;
    const char *flow_out = NULL;
    ptts_params params = PTTS_PARAMS_DEFAULT;

    static struct option long_opts[] = {
        {"dir", required_argument, 0, 'd'},
        {"prompt", required_argument, 0, 'p'},
        {"output", required_argument, 0, 'o'},
        {"voice", required_argument, 0, 0},
        {"info", no_argument, 0, 0},
        {"list", no_argument, 0, 0},
        {"find", required_argument, 0, 0},
        {"verify", no_argument, 0, 0},
        {"tokens", no_argument, 0, 0},
        {"flow-test", no_argument, 0, 0},
        {"mimi-test", no_argument, 0, 0},
        {"mimi-wave", required_argument, 0, 0},
        {"frames", required_argument, 0, 0},
        {"latent-out", required_argument, 0, 0},
        {"cond-out", required_argument, 0, 0},
        {"flow-out", required_argument, 0, 0},
        {"noise-clamp", required_argument, 0, 0},
        {"eos-threshold", required_argument, 0, 0},
        {"eos-min-frames", required_argument, 0, 0},
        {"eos-after", required_argument, 0, 0},
        {"temp", required_argument, 0, 't'},
        {"dummy", no_argument, 0, 0},
        {"rate", required_argument, 0, 'r'},
        {"steps", required_argument, 0, 's'},
        {"seed", required_argument, 0, 'S'},
        {"quiet", no_argument, 0, 'q'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int long_idx = 0;
    while ((opt = getopt_long(argc, argv, "d:p:o:r:s:S:t:qvh", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 0:
                if (strcmp(long_opts[long_idx].name, "info") == 0) info_only = 1;
                else if (strcmp(long_opts[long_idx].name, "voice") == 0) voice = optarg;
                else if (strcmp(long_opts[long_idx].name, "list") == 0) list_tensors = 1;
                else if (strcmp(long_opts[long_idx].name, "find") == 0) find_pat = optarg;
                else if (strcmp(long_opts[long_idx].name, "verify") == 0) verify_weights = 1;
                else if (strcmp(long_opts[long_idx].name, "tokens") == 0) show_tokens = 1;
                else if (strcmp(long_opts[long_idx].name, "flow-test") == 0) flow_test = 1;
                else if (strcmp(long_opts[long_idx].name, "mimi-test") == 0) mimi_test = 1;
                else if (strcmp(long_opts[long_idx].name, "mimi-wave") == 0) mimi_wave = optarg;
                else if (strcmp(long_opts[long_idx].name, "frames") == 0) params.num_frames = atoi(optarg);
                else if (strcmp(long_opts[long_idx].name, "latent-out") == 0) latent_out = optarg;
                else if (strcmp(long_opts[long_idx].name, "cond-out") == 0) cond_out = optarg;
                else if (strcmp(long_opts[long_idx].name, "flow-out") == 0) flow_out = optarg;
                else if (strcmp(long_opts[long_idx].name, "noise-clamp") == 0) params.noise_clamp = (float)atof(optarg);
                else if (strcmp(long_opts[long_idx].name, "eos-threshold") == 0) {
                    params.eos_enabled = 1;
                    params.eos_threshold = (float)atof(optarg);
                } else if (strcmp(long_opts[long_idx].name, "eos-min-frames") == 0) {
                    params.eos_min_frames = atoi(optarg);
                } else if (strcmp(long_opts[long_idx].name, "eos-after") == 0) {
                    params.eos_after = atoi(optarg);
                }
                else if (strcmp(long_opts[long_idx].name, "dummy") == 0) use_dummy = 1;
                break;
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 'r': params.sample_rate = atoi(optarg); break;
            case 's': params.num_steps = atoi(optarg); break;
            case 'S': params.seed = atoll(optarg); break;
            case 't': params.temp = (float)atof(optarg); break;
            case 'q': output_level = OUTPUT_QUIET; break;
            case 'v': output_level = OUTPUT_VERBOSE; break;
            case 'h': print_usage(argv[0]); return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (params.num_frames < 0) params.num_frames = 0;
    if (params.eos_min_frames < 1) params.eos_min_frames = 1;
    if (params.eos_after < 0) params.eos_after = 0;

    if (info_only || list_tensors || show_tokens || find_pat || verify_weights || flow_test || mimi_test || mimi_wave) {
        if (!model_dir) {
            fprintf(stderr, "Error: --dir is required for --info/--list/--find/--tokens/--verify/--flow-test/--mimi-test/--mimi-wave\n");
            return 1;
        }
        ptts_ctx *ctx = ptts_load_dir(model_dir);
        if (!ctx) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            return 1;
        }
        if (info_only) ptts_print_info(ctx);
        if (list_tensors) ptts_list_tensors(ctx);
        if (find_pat) ptts_list_tensors_matching(ctx, find_pat);
        if (verify_weights) {
            int rc = ptts_verify_weights(ctx, output_level >= OUTPUT_VERBOSE);
            if (rc != 0) {
                fprintf(stderr, "Error: weight verification failed\n");
                ptts_free(ctx);
                return 1;
            }
        }
        if (show_tokens) {
            if (!prompt) {
                fprintf(stderr, "Error: --prompt is required for --tokens\n");
                ptts_free(ctx);
                return 1;
            }
            int word_count = 0;
            int eos_after_guess = 0;
            char *prepared = ptts_prepare_text(prompt, &word_count, &eos_after_guess);
            if (!prepared) {
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                ptts_free(ctx);
                return 1;
            }
            int *ids = NULL;
            int n = 0;
            if (ptts_tokenize(ctx, prepared, &ids, &n) != 0) {
                free(prepared);
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                ptts_free(ctx);
                return 1;
            }
            if (output_level >= OUTPUT_VERBOSE) {
                fprintf(stderr, "Prepared text: %s\n", prepared);
            }
            printf("Tokens (%d):", n);
            for (int i = 0; i < n; i++) printf(" %d", ids[i]);
            printf("\n");

            if (output_level >= OUTPUT_VERBOSE) {
                for (int i = 0; i < n; i++) {
                    int plen = 0;
                    const char *piece = ptts_token_piece(ctx, ids[i], &plen);
                    printf("%d: ", ids[i]);
                    if (piece && plen > 0) {
                        for (int j = 0; j < plen; j++) {
                            unsigned char c = (unsigned char)piece[j];
                            if (c >= 32 && c <= 126 && c != '\\') {
                                putchar(c);
                            } else {
                                printf("\\\\x%02X", c);
                            }
                        }
                    }
                    printf("\n");
                }
            }
            free(ids);
            free(prepared);
        }
        if (flow_test || mimi_test || mimi_wave) {
            if (!prompt) {
                fprintf(stderr, "Error: --prompt is required for --flow-test/--mimi-test/--mimi-wave\n");
                ptts_free(ctx);
                return 1;
            }
            int word_count = 0;
            int eos_after_guess = 0;
            char *prepared = ptts_prepare_text(prompt, &word_count, &eos_after_guess);
            if (!prepared) {
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                ptts_free(ctx);
                return 1;
            }
            int *ids = NULL;
            int n = 0;
            if (ptts_tokenize(ctx, prepared, &ids, &n) != 0) {
                free(prepared);
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                ptts_free(ctx);
                return 1;
            }
            if (output_level >= OUTPUT_VERBOSE) {
                fprintf(stderr, "Prepared text: %s\n", prepared);
            }
            float *voice_cond = NULL;
            int voice_len = 0;
            if (ptts_load_voice_conditioning(ctx, voice, &voice_cond, &voice_len) != 0) {
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                free(ids);
                free(prepared);
                ptts_free(ctx);
                return 1;
            }
            ptts_flowlm *fm = ptts_flowlm_load(ctx);
            if (!fm) {
                fprintf(stderr, "Error: failed to load FlowLM weights\n");
                free(voice_cond);
                free(ids);
                free(prepared);
                ptts_free(ctx);
                return 1;
            }
            int gen_frames = params.num_frames;
            if (gen_frames <= 0) {
                gen_frames = (mimi_wave || mimi_test) ? ptts_estimate_frames(word_count) : 1;
            }
            if (params.eos_after <= 0) params.eos_after = eos_after_guess;
            float *latents = (float *)malloc(sizeof(float) * 32 * (size_t)gen_frames);
            if (!latents) {
                fprintf(stderr, "Error: out of memory\n");
                ptts_flowlm_free(fm);
                free(voice_cond);
                free(ids);
                free(prepared);
                ptts_free(ctx);
                return 1;
            }
            float *cond_vec = NULL;
            if (cond_out) {
                cond_vec = (float *)malloc(sizeof(float) * 1024);
                if (!cond_vec) {
                    fprintf(stderr, "Error: out of memory\n");
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
            }
            float *flow_vec = NULL;
            if (flow_out) {
                flow_vec = (float *)malloc(sizeof(float) * 32);
                if (!flow_vec) {
                    fprintf(stderr, "Error: out of memory\n");
                    free(cond_vec);
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
            }
            float eos_first = 0.0f;
            int used_frames = 0;
            if (ptts_flowlm_generate_latents(fm, ids, n, voice_cond, voice_len,
                                             gen_frames, params.num_steps,
                                             params.temp, params.noise_clamp, params.seed,
                                             params.eos_enabled, params.eos_threshold,
                                             params.eos_min_frames, params.eos_after,
                                             latents, &used_frames, &eos_first, cond_vec,
                                             flow_vec) != 0) {
                fprintf(stderr, "Error: flow test failed\n");
                free(flow_vec);
                free(cond_vec);
                free(latents);
                ptts_flowlm_free(fm);
                free(voice_cond);
                free(ids);
                free(prepared);
                ptts_free(ctx);
                return 1;
            }
            float minv = latents[0], maxv = latents[0], sum = 0.0f;
            for (int i = 0; i < 32; i++) {
                if (latents[i] < minv) minv = latents[i];
                if (latents[i] > maxv) maxv = latents[i];
                sum += latents[i];
            }
            printf("FlowLM step: eos_logit=%.4f, latent mean=%.6f min=%.6f max=%.6f\n",
                   eos_first, sum / 32.0f, minv, maxv);
            if (cond_out && cond_vec) {
                FILE *f = fopen(cond_out, "wb");
                if (!f) {
                    fprintf(stderr, "Error: cannot write %s\n", cond_out);
                    free(cond_vec);
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
                fwrite(cond_vec, sizeof(float), 1024, f);
                fclose(f);
                if (output_level >= OUTPUT_VERBOSE) {
                    fprintf(stderr, "Wrote FlowLM cond to %s\n", cond_out);
                }
            }
            if (flow_out && flow_vec) {
                FILE *f = fopen(flow_out, "wb");
                if (!f) {
                    fprintf(stderr, "Error: cannot write %s\n", flow_out);
                    free(flow_vec);
                    free(cond_vec);
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
                fwrite(flow_vec, sizeof(float), 32, f);
                fclose(f);
                if (output_level >= OUTPUT_VERBOSE) {
                    fprintf(stderr, "Wrote FlowLM flow to %s\n", flow_out);
                }
            }
            if (latent_out) {
                FILE *f = fopen(latent_out, "wb");
                if (!f) {
                    fprintf(stderr, "Error: cannot write %s\n", latent_out);
                    free(flow_vec);
                    free(cond_vec);
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
                fwrite(latents, sizeof(float), (size_t)used_frames * 32, f);
                fclose(f);
                if (output_level >= OUTPUT_VERBOSE) {
                    fprintf(stderr, "Wrote %d latent frame(s) to %s\n", used_frames, latent_out);
                }
            }
            if (mimi_test || mimi_wave) {
                ptts_mimi *mm = ptts_mimi_load(ctx);
                if (!mm) {
                    fprintf(stderr, "Error: failed to load Mimi weights\n");
                    free(flow_vec);
                    free(cond_vec);
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
                float *scaled = (float *)malloc(sizeof(float) * 32 * (size_t)used_frames);
                if (!scaled) {
                    fprintf(stderr, "Error: out of memory\n");
                    ptts_mimi_free(mm);
                    free(flow_vec);
                    free(cond_vec);
                    free(latents);
                    ptts_flowlm_free(fm);
                    free(voice_cond);
                    free(ids);
                    free(prepared);
                    ptts_free(ctx);
                    return 1;
                }
                ptts_flowlm_scale_latents(fm, latents, used_frames, scaled);
                if (mimi_test) {
                    float embed[512];
                    if (ptts_mimi_forward_one(mm, scaled, embed) != 0) {
                        fprintf(stderr, "Error: Mimi test failed\n");
                        free(scaled);
                        ptts_mimi_free(mm);
                        free(flow_vec);
                        free(cond_vec);
                        free(latents);
                        ptts_flowlm_free(fm);
                        free(voice_cond);
                        free(ids);
                        free(prepared);
                        ptts_free(ctx);
                        return 1;
                    }
                    float emin = embed[0], emax = embed[0], esum = 0.0f;
                    for (int i = 0; i < 512; i++) {
                        if (embed[i] < emin) emin = embed[i];
                        if (embed[i] > emax) emax = embed[i];
                        esum += embed[i];
                    }
                    printf("Mimi decode (transformer) stats: mean=%.6f min=%.6f max=%.6f\n",
                           esum / 512.0f, emin, emax);
                }
                if (mimi_wave) {
                    const int frame_samples = 16 * 6 * 5 * 4;
                    int total_samples = frame_samples * used_frames;
                    float *all = (float *)malloc(sizeof(float) * (size_t)total_samples);
                    if (!all) {
                        fprintf(stderr, "Error: out of memory\n");
                        free(scaled);
                        ptts_mimi_free(mm);
                        free(flow_vec);
                        free(cond_vec);
                        free(latents);
                        ptts_flowlm_free(fm);
                        free(voice_cond);
                        free(ids);
                        free(prepared);
                        ptts_free(ctx);
                        return 1;
                    }
                    int wav_len = 0;
                    if (ptts_mimi_decode(mm, scaled, used_frames, all, &wav_len) != 0) {
                        fprintf(stderr, "Error: Mimi wave decode failed\n");
                        free(all);
                        free(scaled);
                        ptts_mimi_free(mm);
                        free(flow_vec);
                        free(cond_vec);
                        free(latents);
                        ptts_flowlm_free(fm);
                        free(voice_cond);
                        free(ids);
                        free(prepared);
                        ptts_free(ctx);
                        return 1;
                    }
                    if (wav_len != total_samples) {
                        fprintf(stderr, "Error: unexpected Mimi length (%d samples)\n", wav_len);
                        free(all);
                        free(scaled);
                        ptts_mimi_free(mm);
                        free(flow_vec);
                        free(cond_vec);
                        free(latents);
                        ptts_flowlm_free(fm);
                        free(voice_cond);
                        free(ids);
                        free(prepared);
                        ptts_free(ctx);
                        return 1;
                    }
                    ptts_audio *audio = ptts_audio_create(PTTS_DEFAULT_SAMPLE_RATE, 1, total_samples);
                    if (!audio) {
                        fprintf(stderr, "Error: failed to allocate audio buffer\n");
                        free(all);
                        free(scaled);
                        ptts_mimi_free(mm);
                        free(flow_vec);
                        free(cond_vec);
                        free(latents);
                        ptts_flowlm_free(fm);
                        free(voice_cond);
                        free(ids);
                        free(prepared);
                        ptts_free(ctx);
                        return 1;
                    }
                    memcpy(audio->samples, all, (size_t)total_samples * sizeof(float));
                    free(all);
                    if (ptts_audio_save_wav(audio, mimi_wave) != 0) {
                        fprintf(stderr, "Error: failed to write WAV\n");
                        ptts_audio_free(audio);
                        free(scaled);
                        ptts_mimi_free(mm);
                        free(flow_vec);
                        free(cond_vec);
                        free(latents);
                        ptts_flowlm_free(fm);
                        free(voice_cond);
                        free(ids);
                        free(prepared);
                        ptts_free(ctx);
                        return 1;
                    }
                    ptts_audio_free(audio);
                    if (output_level >= OUTPUT_VERBOSE) {
                        fprintf(stderr, "Wrote Mimi WAV to %s (%d frames, %d samples)\n",
                                mimi_wave, used_frames, frame_samples * used_frames);
                    }
                }
                free(scaled);
                ptts_mimi_free(mm);
            }
            free(flow_vec);
            free(cond_vec);
            free(latents);
            ptts_flowlm_free(fm);
            free(voice_cond);
            free(ids);
            free(prepared);
        }
        ptts_free(ctx);
        return 0;
    }

    if (!prompt) {
        fprintf(stderr, "Error: --prompt is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!output) {
        fprintf(stderr, "Error: --output is required\n");
        print_usage(argv[0]);
        return 1;
    }

    ptts_audio *audio = NULL;

    if (use_dummy) {
        LOG_NORMAL("Generating dummy audio...\n");
        audio = ptts_generate_dummy(prompt, &params);
    } else {
        if (!model_dir) {
            fprintf(stderr, "Error: --dir is required unless --dummy is used\n");
            return 1;
        }
        ptts_ctx *ctx = ptts_load_dir(model_dir);
        if (!ctx) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            return 1;
        }
        LOG_VERBOSE("Loaded model, starting inference...\n");
        audio = ptts_generate(ctx, prompt, voice, &params);
        if (!audio) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            ptts_free(ctx);
            return 1;
        }
        ptts_free(ctx);
    }

    if (!audio) {
        fprintf(stderr, "Error: %s\n", ptts_get_error());
        return 1;
    }

    if (ptts_audio_save_wav(audio, output) != 0) {
        fprintf(stderr, "Error: failed to write WAV\n");
        ptts_audio_free(audio);
        return 1;
    }

    ptts_audio_free(audio);
    LOG_NORMAL("Saved %s\n", output);
    return 0;
}
