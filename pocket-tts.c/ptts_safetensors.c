/*
 * ptts_safetensors.c - Safetensors file format reader implementation
 */

#include "ptts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* Minimal JSON parser for safetensors header */

static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_string(const char **p, char *out, size_t max_len) {
    skip_whitespace(p);
    if (**p != '"') return -1;
    (*p)++;

    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == 'r') out[i++] = '\r';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else out[i++] = **p;
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';

    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

static int64_t parse_int(const char **p) {
    skip_whitespace(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

static safetensor_dtype_t parse_dtype(const char *s) {
    if (strcmp(s, "F32") == 0) return DTYPE_F32;
    if (strcmp(s, "F16") == 0) return DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "I32") == 0) return DTYPE_I32;
    if (strcmp(s, "I64") == 0) return DTYPE_I64;
    if (strcmp(s, "BOOL") == 0) return DTYPE_BOOL;
    return DTYPE_UNKNOWN;
}

/* Parse a tensor entry from JSON */
static int parse_tensor_entry(const char **p, safetensor_t *t) {
    skip_whitespace(p);
    if (**p != '{') return -1;
    (*p)++;

    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_whitespace(p);
        if (**p == ',') { (*p)++; continue; }

        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0) return -1;

        skip_whitespace(p);
        if (**p != ':') return -1;
        (*p)++;
        skip_whitespace(p);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (parse_string(p, dtype_str, sizeof(dtype_str)) != 0) return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']' && t->ndim < 8) {
                skip_whitespace(p);
                if (**p == ',') { (*p)++; continue; }
                t->shape[t->ndim++] = parse_int(p);
            }
            if (**p == ']') (*p)++;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            skip_whitespace(p);
            size_t start = (size_t)parse_int(p);
            skip_whitespace(p);
            if (**p == ',') (*p)++;
            skip_whitespace(p);
            size_t end = (size_t)parse_int(p);
            t->data_offset = start;
            t->data_size = end - start;
            skip_whitespace(p);
            if (**p == ']') (*p)++;
        } else {
            /* Skip unknown value */
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\') (*p)++;
                    if (**p) (*p)++;
                }
                if (**p == '"') (*p)++;
            } else if (**p == '[') {
                int depth = 1;
                (*p)++;
                while (**p && depth > 0) {
                    if (**p == '[') depth++;
                    else if (**p == ']') depth--;
                    (*p)++;
                }
            } else if (**p == '{') {
                int depth = 1;
                (*p)++;
                while (**p && depth > 0) {
                    if (**p == '{') depth++;
                    else if (**p == '}') depth--;
                    (*p)++;
                }
            } else {
                while (**p && **p != ',' && **p != '}') (*p)++;
            }
        }
    }

    if (**p == '}') (*p)++;
    return 0;
}

/* Parse the entire JSON header */
static int parse_header(safetensors_file_t *sf) {
    const char *p = sf->header_json;
    skip_whitespace(&p);

    if (*p != '{') return -1;
    p++;

    sf->num_tensors = 0;

    while (*p && *p != '}' && sf->num_tensors < SAFETENSORS_MAX_TENSORS) {
        skip_whitespace(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        /* Parse tensor name */
        char name[256];
        if (parse_string(&p, name, sizeof(name)) != 0) return -1;

        skip_whitespace(&p);
        if (*p != ':') return -1;
        p++;

        /* Skip __metadata__ entry */
        if (strcmp(name, "__metadata__") == 0) {
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;
        }

        /* Parse tensor entry */
        safetensor_t *t = &sf->tensors[sf->num_tensors];
        snprintf(t->name, sizeof(t->name), "%s", name);

        if (parse_tensor_entry(&p, t) != 0) return -1;
        sf->num_tensors++;
    }

    return 0;
}

safetensors_file_t *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return NULL;

    safetensors_file_t *sf = (safetensors_file_t *)calloc(1, sizeof(safetensors_file_t));
    if (!sf) {
        munmap(data, st.st_size);
        return NULL;
    }

    sf->path = strdup(path);
    sf->data = data;
    sf->file_size = st.st_size;

    /* Read header size */
    if (sf->file_size < 8) {
        safetensors_close(sf);
        return NULL;
    }

    uint64_t header_size = 0;
    memcpy(&header_size, data, 8);
    sf->header_size = (size_t)header_size;

    if (sf->header_size + 8 > sf->file_size) {
        safetensors_close(sf);
        return NULL;
    }

    sf->header_json = (char *)malloc(sf->header_size + 1);
    if (!sf->header_json) {
        safetensors_close(sf);
        return NULL;
    }
    memcpy(sf->header_json, (char *)data + 8, sf->header_size);
    sf->header_json[sf->header_size] = '\0';

    if (parse_header(sf) != 0) {
        safetensors_close(sf);
        return NULL;
    }

    return sf;
}

void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;
    if (sf->data) munmap(sf->data, sf->file_size);
    free(sf->header_json);
    free(sf->path);
    free(sf);
}

const safetensor_t *safetensors_find(const safetensors_file_t *sf, const char *name) {
    if (!sf || !name) return NULL;
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return &sf->tensors[i];
    }
    return NULL;
}

const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t) return NULL;
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t) return NULL;

    int64_t numel = safetensor_numel(t);
    if (numel <= 0) return NULL;

    float *out = (float *)malloc(numel * sizeof(float));
    if (!out) return NULL;

    const void *src = safetensors_data(sf, t);
    if (!src) {
        free(out);
        return NULL;
    }

    if (t->dtype == DTYPE_F32) {
        memcpy(out, src, numel * sizeof(float));
    } else if (t->dtype == DTYPE_F16) {
        const uint16_t *in = (const uint16_t *)src;
        for (int64_t i = 0; i < numel; i++) {
            uint16_t h = in[i];
            uint16_t sign = (h >> 15) & 1;
            uint16_t exp = (h >> 10) & 0x1f;
            uint16_t mant = h & 0x3ff;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    f = (uint32_t)sign << 31;
                } else {
                    exp = 1;
                    while ((mant & 0x400) == 0) {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x3ff;
                    exp += 127 - 15;
                    f = ((uint32_t)sign << 31) | ((uint32_t)exp << 23) | ((uint32_t)mant << 13);
                }
            } else if (exp == 31) {
                f = ((uint32_t)sign << 31) | 0x7f800000 | ((uint32_t)mant << 13);
            } else {
                exp += 127 - 15;
                f = ((uint32_t)sign << 31) | ((uint32_t)exp << 23) | ((uint32_t)mant << 13);
            }
            memcpy(&out[i], &f, sizeof(float));
        }
    } else if (t->dtype == DTYPE_BF16) {
        const uint16_t *in = (const uint16_t *)src;
        for (int64_t i = 0; i < numel; i++) {
            uint32_t f = ((uint32_t)in[i]) << 16;
            memcpy(&out[i], &f, sizeof(float));
        }
    } else {
        free(out);
        return NULL;
    }

    return out;
}

uint16_t *safetensors_get_bf16(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t || t->dtype != DTYPE_BF16) return NULL;
    int64_t numel = safetensor_numel(t);
    if (numel <= 0) return NULL;

    uint16_t *out = (uint16_t *)malloc(numel * sizeof(uint16_t));
    if (!out) return NULL;

    const void *src = safetensors_data(sf, t);
    if (!src) {
        free(out);
        return NULL;
    }

    memcpy(out, src, numel * sizeof(uint16_t));
    return out;
}

uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t || t->dtype != DTYPE_BF16) return NULL;
    return (uint16_t *)safetensors_data(sf, t);
}

int safetensor_is_bf16(const safetensor_t *t) {
    return t && t->dtype == DTYPE_BF16;
}

int64_t safetensor_numel(const safetensor_t *t) {
    if (!t) return 0;
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

void safetensor_print(const safetensor_t *t) {
    if (!t) return;
    printf("%s  ", t->name);
    printf("[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld", (long long)t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("]  ");
    switch (t->dtype) {
        case DTYPE_F32: printf("F32"); break;
        case DTYPE_F16: printf("F16"); break;
        case DTYPE_BF16: printf("BF16"); break;
        case DTYPE_I32: printf("I32"); break;
        case DTYPE_I64: printf("I64"); break;
        case DTYPE_BOOL: printf("BOOL"); break;
        default: printf("UNKNOWN"); break;
    }
    printf("\n");
}

void safetensors_print_all(const safetensors_file_t *sf) {
    if (!sf) return;
    printf("Tensors: %d\n", sf->num_tensors);
    for (int i = 0; i < sf->num_tensors; i++) {
        safetensor_print(&sf->tensors[i]);
    }
}
