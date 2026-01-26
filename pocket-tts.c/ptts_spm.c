/*
 * Minimal SentencePiece tokenizer (WIP)
 *
 * This implementation focuses on loading the ModelProto and doing
 * unigram-style Viterbi segmentation with piece scores. Normalization
 * uses SentencePiece precompiled charsmap when available.
 */

#include "ptts_spm.h"
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ptts_spm_piece {
    char *bytes;
    int len;
    float score;
    int type;
};

struct ptts_spm {
    struct ptts_spm_piece *pieces;
    int num_pieces;
    int cap_pieces;
    int unk_id;
    int max_piece_len;
    int add_dummy_prefix;
    int remove_extra_whitespaces;
    int escape_whitespaces;
    int treat_whitespace_as_suffix;
    uint8_t *charsmap;
    size_t charmap_len;
    const uint32_t *xcda_array;
    size_t xcda_array_size;
    const char *prefix_replacements;
    size_t prefix_replacements_size;
    char **user_pieces;
    int *user_pieces_len;
    int num_user_pieces;
    int cap_user_pieces;
};

/* ========================================================================
 * Protobuf parsing helpers
 * ======================================================================== */

static int read_varint(const uint8_t **p, const uint8_t *end, uint64_t *out) {
    uint64_t val = 0;
    int shift = 0;
    while (*p < end && shift < 64) {
        uint8_t b = **p;
        (*p)++;
        val |= ((uint64_t)(b & 0x7f)) << shift;
        if ((b & 0x80) == 0) {
            *out = val;
            return 0;
        }
        shift += 7;
    }
    return -1;
}

static int skip_field(int wire, const uint8_t **p, const uint8_t *end) {
    uint64_t len = 0;
    switch (wire) {
        case 0: /* varint */
            return read_varint(p, end, &len);
        case 1: /* 64-bit */
            if ((size_t)(end - *p) < 8) return -1;
            *p += 8;
            return 0;
        case 2: /* length-delimited */
            if (read_varint(p, end, &len) != 0) return -1;
            if ((size_t)(end - *p) < len) return -1;
            *p += len;
            return 0;
        case 5: /* 32-bit */
            if ((size_t)(end - *p) < 4) return -1;
            *p += 4;
            return 0;
        default:
            return -1;
    }
}

static float read_fixed32_as_float(const uint8_t *p) {
    uint32_t u = 0;
    memcpy(&u, p, 4);
    float f = 0.0f;
    memcpy(&f, &u, 4);
    return f;
}

static int parse_sentence_piece(const uint8_t *buf, size_t len, struct ptts_spm_piece *out) {
    const uint8_t *p = buf;
    const uint8_t *end = buf + len;

    out->bytes = NULL;
    out->len = 0;
    out->score = 0.0f;
    out->type = 0;

    while (p < end) {
        uint64_t key = 0;
        if (read_varint(&p, end, &key) != 0) {
            free(out->bytes);
            out->bytes = NULL;
            return -1;
        }
        int field = (int)(key >> 3);
        int wire = (int)(key & 0x7);

        if (field == 1 && wire == 2) {
            uint64_t slen = 0;
            if (read_varint(&p, end, &slen) != 0 || (size_t)(end - p) < slen) {
                free(out->bytes);
                out->bytes = NULL;
                return -1;
            }
            out->bytes = (char *)malloc((size_t)slen + 1);
            if (!out->bytes) return -1;
            memcpy(out->bytes, p, (size_t)slen);
            out->bytes[slen] = '\0';
            out->len = (int)slen;
            p += slen;
        } else if (field == 2 && wire == 5) {
            if ((size_t)(end - p) < 4) {
                free(out->bytes);
                out->bytes = NULL;
                return -1;
            }
            out->score = read_fixed32_as_float(p);
            p += 4;
        } else if (field == 3 && wire == 0) {
            uint64_t v = 0;
            if (read_varint(&p, end, &v) != 0) {
                free(out->bytes);
                out->bytes = NULL;
                return -1;
            }
            out->type = (int)v;
        } else {
            if (skip_field(wire, &p, end) != 0) {
                free(out->bytes);
                out->bytes = NULL;
                return -1;
            }
        }
    }

    return 0;
}

static void spm_set_defaults(ptts_spm *spm) {
    spm->add_dummy_prefix = 1;
    spm->remove_extra_whitespaces = 1;
    spm->escape_whitespaces = 1;
    spm->treat_whitespace_as_suffix = 0;
}

static int parse_normalizer_spec(const uint8_t *buf, size_t len, ptts_spm *spm) {
    const uint8_t *p = buf;
    const uint8_t *end = buf + len;
    while (p < end) {
        uint64_t key = 0;
        if (read_varint(&p, end, &key) != 0) return -1;
        int field = (int)(key >> 3);
        int wire = (int)(key & 0x7);

        if (field == 2 && wire == 2) { /* precompiled_charsmap */
            uint64_t blen = 0;
            if (read_varint(&p, end, &blen) != 0) return -1;
            if ((size_t)(end - p) < blen) return -1;
            free(spm->charsmap);
            spm->charsmap = (uint8_t *)malloc((size_t)blen);
            if (!spm->charsmap) return -1;
            memcpy(spm->charsmap, p, (size_t)blen);
            spm->charmap_len = (size_t)blen;
            p += blen;
        } else if (field == 3 && wire == 0) { /* add_dummy_prefix */
            uint64_t v = 0;
            if (read_varint(&p, end, &v) != 0) return -1;
            spm->add_dummy_prefix = (v != 0);
        } else if (field == 4 && wire == 0) { /* remove_extra_whitespaces */
            uint64_t v = 0;
            if (read_varint(&p, end, &v) != 0) return -1;
            spm->remove_extra_whitespaces = (v != 0);
        } else if (field == 5 && wire == 0) { /* escape_whitespaces */
            uint64_t v = 0;
            if (read_varint(&p, end, &v) != 0) return -1;
            spm->escape_whitespaces = (v != 0);
        } else {
            if (skip_field(wire, &p, end) != 0) return -1;
        }
    }
    return 0;
}

static int parse_trainer_spec(const uint8_t *buf, size_t len, ptts_spm *spm) {
    const uint8_t *p = buf;
    const uint8_t *end = buf + len;
    while (p < end) {
        uint64_t key = 0;
        if (read_varint(&p, end, &key) != 0) return -1;
        int field = (int)(key >> 3);
        int wire = (int)(key & 0x7);

        if (field == 24 && wire == 0) { /* treat_whitespace_as_suffix */
            uint64_t v = 0;
            if (read_varint(&p, end, &v) != 0) return -1;
            spm->treat_whitespace_as_suffix = (v != 0);
        } else {
            if (skip_field(wire, &p, end) != 0) return -1;
        }
    }
    return 0;
}

/* ========================================================================
 * UTF-8 helpers
 * ======================================================================== */

static int is_utf8_lead(unsigned char c) {
    return (c & 0xC0) != 0x80;
}

static int boundary_index_for_offset(const int *pos, int n, int offset) {
    int lo = 0, hi = n;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int v = pos[mid];
        if (v == offset) return mid;
        if (v < offset) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ========================================================================
 * Normalization (SentencePiece)
 * ======================================================================== */

static void spm_add_user_piece(ptts_spm *spm, const char *piece, int len) {
    if (!spm || !piece || len <= 0) return;
    if (spm->num_user_pieces >= spm->cap_user_pieces) {
        int new_cap = spm->cap_user_pieces ? spm->cap_user_pieces * 2 : 32;
        char **np = (char **)realloc(spm->user_pieces, (size_t)new_cap * sizeof(char *));
        int *nl = (int *)realloc(spm->user_pieces_len, (size_t)new_cap * sizeof(int));
        if (!np || !nl) return;
        spm->user_pieces = np;
        spm->user_pieces_len = nl;
        spm->cap_user_pieces = new_cap;
    }
    spm->user_pieces[spm->num_user_pieces] = (char *)piece;
    spm->user_pieces_len[spm->num_user_pieces] = len;
    spm->num_user_pieces++;
}

static void spm_setup_charsmap(ptts_spm *spm) {
    spm->xcda_array = NULL;
    spm->xcda_array_size = 0;
    spm->prefix_replacements = NULL;
    spm->prefix_replacements_size = 0;

    if (!spm->charsmap || spm->charmap_len < 4) return;
    uint32_t blob_size = 0;
    memcpy(&blob_size, spm->charsmap, sizeof(uint32_t));
    size_t offset = sizeof(uint32_t);
    if (offset + blob_size > spm->charmap_len) return;
    if (blob_size % sizeof(uint32_t) != 0) return;

    spm->xcda_array = (const uint32_t *)(spm->charsmap + offset);
    spm->xcda_array_size = blob_size / sizeof(uint32_t);
    offset += blob_size;
    spm->prefix_replacements = (const char *)(spm->charsmap + offset);
    spm->prefix_replacements_size = spm->charmap_len - offset;
}

static int utf8_decode_len(const char *s, size_t avail, size_t *out_len) {
    if (avail == 0) return -1;
    unsigned char c0 = (unsigned char)s[0];
    if (c0 < 0x80) {
        *out_len = 1;
        return 0;
    }
    if (c0 < 0xC2) return -1;
    if (c0 < 0xE0) {
        if (avail < 2) return -1;
        unsigned char c1 = (unsigned char)s[1];
        if ((c1 & 0xC0) != 0x80) return -1;
        *out_len = 2;
        return 0;
    }
    if (c0 < 0xF0) {
        if (avail < 3) return -1;
        unsigned char c1 = (unsigned char)s[1];
        unsigned char c2 = (unsigned char)s[2];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return -1;
        if (c0 == 0xE0 && c1 < 0xA0) return -1; /* overlong */
        if (c0 == 0xED && c1 >= 0xA0) return -1; /* surrogate */
        *out_len = 3;
        return 0;
    }
    if (c0 < 0xF5) {
        if (avail < 4) return -1;
        unsigned char c1 = (unsigned char)s[1];
        unsigned char c2 = (unsigned char)s[2];
        unsigned char c3 = (unsigned char)s[3];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return -1;
        if (c0 == 0xF0 && c1 < 0x90) return -1; /* overlong */
        if (c0 == 0xF4 && c1 > 0x8F) return -1; /* > U+10FFFF */
        *out_len = 4;
        return 0;
    }
    return -1;
}

static uint32_t xcda_get_base(const ptts_spm *spm, uint32_t index) {
    uint32_t node = spm->xcda_array[index];
    return (node >> 10) << ((node & (1U << 9)) >> 6);
}

static uint32_t xcda_get_lcheck(const ptts_spm *spm, uint32_t index) {
    uint32_t node = spm->xcda_array[index];
    return node & ((1U << 31) | 0xff);
}

static int xcda_get_leaf(const ptts_spm *spm, uint32_t index) {
    uint32_t node = spm->xcda_array[index];
    return (node >> 8) & 1U;
}

static uint32_t xcda_get_value(const ptts_spm *spm, uint32_t index) {
    uint32_t node = spm->xcda_array[index];
    return node & ((1U << 31) - 1);
}

struct norm_result {
    const char *normalized;
    size_t normalized_len;
    size_t consumed_input;
};

static size_t spm_user_defined_match(const ptts_spm *spm, const char *input, size_t input_len, size_t input_offset) {
    size_t best = 0;
    for (int i = 0; i < spm->num_user_pieces; i++) {
        int plen = spm->user_pieces_len[i];
        if ((size_t)plen > input_len - input_offset) continue;
        if (memcmp(input + input_offset, spm->user_pieces[i], (size_t)plen) == 0) {
            if ((size_t)plen > best) best = (size_t)plen;
        }
    }
    return best;
}

static struct norm_result spm_normalize_prefix(const ptts_spm *spm, const char *input, size_t input_len, size_t input_offset) {
    if (input_offset >= input_len) {
        return (struct norm_result){ input + input_offset, 0, 0 };
    }

    size_t user_match = spm_user_defined_match(spm, input, input_len, input_offset);
    if (user_match > 0) {
        return (struct norm_result){ input + input_offset, user_match, user_match };
    }

    size_t longest_prefix_length = 0;
    size_t longest_prefix_offset = 0;

    if (spm->xcda_array_size > 0) {
        uint32_t node_index = 0;
        if (node_index >= spm->xcda_array_size) {
            return (struct norm_result){ input + input_offset, 1, 1 };
        }
        node_index = xcda_get_base(spm, node_index);
        for (size_t prefix_offset = input_offset; prefix_offset < input_len; prefix_offset++) {
            unsigned char c = (unsigned char)input[prefix_offset];
            if (c == 0) break;
            node_index ^= c;
            if (node_index >= spm->xcda_array_size) break;
            if (xcda_get_lcheck(spm, node_index) != c) break;
            int is_leaf = xcda_get_leaf(spm, node_index);
            node_index ^= xcda_get_base(spm, node_index);
            if (node_index >= spm->xcda_array_size) break;
            if (is_leaf) {
                longest_prefix_length = prefix_offset - input_offset + 1;
                longest_prefix_offset = xcda_get_value(spm, node_index);
            }
        }
    }

    if (longest_prefix_length > 0) {
        if (longest_prefix_offset >= spm->prefix_replacements_size) {
            return (struct norm_result){ input + input_offset, 1, 1 };
        }
        const char *rep = spm->prefix_replacements + longest_prefix_offset;
        return (struct norm_result){ rep, strlen(rep), longest_prefix_length };
    }

    size_t len = 0;
    if (utf8_decode_len(input + input_offset, input_len - input_offset, &len) == 0) {
        return (struct norm_result){ input + input_offset, len, len };
    }

    return (struct norm_result){ "\xEF\xBF\xBD", 3, 1 };
}

static int spm_append(char **out, size_t *out_len, size_t *cap, const char *s, size_t n) {
    if (*out_len + n + 1 > *cap) {
        size_t new_cap = (*cap) ? *cap : 64;
        while (*out_len + n + 1 > new_cap) new_cap *= 2;
        char *np = (char *)realloc(*out, new_cap);
        if (!np) return -1;
        *out = np;
        *cap = new_cap;
    }
    memcpy(*out + *out_len, s, n);
    *out_len += n;
    (*out)[*out_len] = '\0';
    return 0;
}

static char *spm_normalize(const ptts_spm *spm, const char *text) {
    if (!text) return NULL;
    size_t in_len = strlen(text);
    if (in_len == 0) {
        char *empty = (char *)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    const char *space = spm->escape_whitespaces ? "\xE2\x96\x81" : " ";
    const size_t space_len = spm->escape_whitespaces ? 3 : 1;

    const int shall_prepend_space = (!spm->treat_whitespace_as_suffix) && spm->add_dummy_prefix;
    const int shall_append_space = (spm->treat_whitespace_as_suffix) && spm->add_dummy_prefix;
    const int shall_merge_spaces = spm->remove_extra_whitespaces;

    int is_space_prepended = 0;
    int processing_non_ws = 0;

    char *out = NULL;
    size_t out_len = 0;
    size_t cap = 0;

    for (size_t input_offset = 0; input_offset < in_len; ) {
        struct norm_result res = spm_normalize_prefix(spm, text, in_len, input_offset);
        for (size_t i = 0; i < res.normalized_len; i++) {
            char c = res.normalized[i];
            if (c != ' ') {
                if (!processing_non_ws) {
                    processing_non_ws = 1;
                    if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                        if (spm_append(&out, &out_len, &cap, space, space_len) != 0) {
                            free(out);
                            return NULL;
                        }
                        is_space_prepended = 1;
                    }
                }
                if (spm_append(&out, &out_len, &cap, &c, 1) != 0) {
                    free(out);
                    return NULL;
                }
            } else {
                if (processing_non_ws) processing_non_ws = 0;
                if (!shall_merge_spaces) {
                    if (spm_append(&out, &out_len, &cap, space, space_len) != 0) {
                        free(out);
                        return NULL;
                    }
                }
            }
        }
        input_offset += res.consumed_input;
    }

    if (shall_append_space) {
        if (spm_append(&out, &out_len, &cap, space, space_len) != 0) {
            free(out);
            return NULL;
        }
    }

    if (!out) {
        out = (char *)malloc(1);
        if (!out) return NULL;
        out[0] = '\0';
    }
    return out;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

ptts_spm *ptts_spm_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return NULL;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)size);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    ptts_spm *spm = (ptts_spm *)calloc(1, sizeof(ptts_spm));
    if (!spm) {
        free(buf);
        return NULL;
    }

    spm->unk_id = -1;
    spm_set_defaults(spm);

    const uint8_t *p = buf;
    const uint8_t *end = buf + size;
    while (p < end) {
        uint64_t key = 0;
        if (read_varint(&p, end, &key) != 0) break;
        int field = (int)(key >> 3);
        int wire = (int)(key & 0x7);

        if (field == 1 && wire == 2) {
            uint64_t msg_len = 0;
            if (read_varint(&p, end, &msg_len) != 0) break;
            if ((size_t)(end - p) < msg_len) break;

            if (spm->num_pieces >= spm->cap_pieces) {
                int new_cap = spm->cap_pieces ? spm->cap_pieces * 2 : 512;
                struct ptts_spm_piece *np = (struct ptts_spm_piece *)realloc(
                    spm->pieces, (size_t)new_cap * sizeof(struct ptts_spm_piece));
                if (!np) break;
                spm->pieces = np;
                spm->cap_pieces = new_cap;
            }

            struct ptts_spm_piece *piece = &spm->pieces[spm->num_pieces];
            if (parse_sentence_piece(p, (size_t)msg_len, piece) != 0) break;

            if (piece->len > spm->max_piece_len) spm->max_piece_len = piece->len;
            if (piece->type == 2 || (piece->bytes && strcmp(piece->bytes, "<unk>") == 0)) {
                spm->unk_id = spm->num_pieces;
            }
            if (piece->type == 4 && piece->bytes) {
                spm_add_user_piece(spm, piece->bytes, piece->len);
            }

            spm->num_pieces++;
            p += msg_len;
        } else if (field == 2 && wire == 2) {
            uint64_t msg_len = 0;
            if (read_varint(&p, end, &msg_len) != 0) break;
            if ((size_t)(end - p) < msg_len) break;
            if (parse_trainer_spec(p, (size_t)msg_len, spm) != 0) break;
            p += msg_len;
        } else if (field == 3 && wire == 2) {
            uint64_t msg_len = 0;
            if (read_varint(&p, end, &msg_len) != 0) break;
            if ((size_t)(end - p) < msg_len) break;
            if (parse_normalizer_spec(p, (size_t)msg_len, spm) != 0) break;
            p += msg_len;
        } else {
            if (skip_field(wire, &p, end) != 0) break;
        }
    }

    free(buf);
    if (spm->num_pieces == 0) {
        ptts_spm_free(spm);
        return NULL;
    }
    spm_setup_charsmap(spm);
    return spm;
}

void ptts_spm_free(ptts_spm *spm) {
    if (!spm) return;
    for (int i = 0; i < spm->num_pieces; i++) {
        free(spm->pieces[i].bytes);
    }
    free(spm->pieces);
    free(spm->charsmap);
    free(spm->user_pieces);
    free(spm->user_pieces_len);
    free(spm);
}

const char *ptts_spm_piece(const ptts_spm *spm, int id, int *out_len) {
    if (!spm || id < 0 || id >= spm->num_pieces) return NULL;
    if (out_len) *out_len = spm->pieces[id].len;
    return spm->pieces[id].bytes;
}

int ptts_spm_vocab_size(const ptts_spm *spm) {
    if (!spm) return 0;
    return spm->num_pieces;
}

int ptts_spm_encode(const ptts_spm *spm, const char *text, int **out_ids, int *out_len) {
    if (!spm || !text || !out_ids || !out_len) return -1;

    char *norm = spm_normalize(spm, text);
    if (!norm) return -1;

    int norm_len = (int)strlen(norm);
    if (norm_len == 0) {
        *out_ids = NULL;
        *out_len = 0;
        free(norm);
        return 0;
    }

    int *pos = (int *)malloc((size_t)(norm_len + 2) * sizeof(int));
    if (!pos) {
        free(norm);
        return -1;
    }

    int n_pos = 0;
    for (int i = 0; i < norm_len; i++) {
        if (is_utf8_lead((unsigned char)norm[i])) {
            pos[n_pos++] = i;
        }
    }
    pos[n_pos++] = norm_len;

    const float NEG = -1e30f;
    float *dp = (float *)malloc((size_t)n_pos * sizeof(float));
    int *prev = (int *)malloc((size_t)n_pos * sizeof(int));
    int *best_id = (int *)malloc((size_t)n_pos * sizeof(int));
    if (!dp || !prev || !best_id) {
        free(norm);
        free(pos);
        free(dp);
        free(prev);
        free(best_id);
        return -1;
    }

    for (int i = 0; i < n_pos; i++) {
        dp[i] = NEG;
        prev[i] = -1;
        best_id[i] = -1;
    }
    dp[0] = 0.0f;

    for (int i = 0; i < n_pos - 1; i++) {
        if (dp[i] <= NEG / 2) continue;
        int matched = 0;
        int start = pos[i];

        for (int j = 0; j < spm->num_pieces; j++) {
            struct ptts_spm_piece *piece = &spm->pieces[j];
            if (!piece->bytes || piece->len <= 0) continue;
            int end = start + piece->len;
            if (end > norm_len) continue;
            if (memcmp(norm + start, piece->bytes, (size_t)piece->len) != 0) continue;

            int end_idx = boundary_index_for_offset(pos, n_pos - 1, end);
            if (end_idx < 0) continue;

            float score = dp[i] + piece->score;
            if (score > dp[end_idx]) {
                dp[end_idx] = score;
                prev[end_idx] = i;
                best_id[end_idx] = j;
            }
            matched = 1;
        }

        if (!matched && spm->unk_id >= 0) {
            int end_idx = i + 1;
            float score = dp[i] + spm->pieces[spm->unk_id].score;
            if (score > dp[end_idx]) {
                dp[end_idx] = score;
                prev[end_idx] = i;
                best_id[end_idx] = spm->unk_id;
            }
        }
    }

    int end_idx = n_pos - 1;
    if (prev[end_idx] < 0) {
        free(norm);
        free(pos);
        free(dp);
        free(prev);
        free(best_id);
        return -1;
    }

    int count = 0;
    for (int i = end_idx; i > 0; i = prev[i]) count++;

    int *ids = (int *)malloc((size_t)count * sizeof(int));
    if (!ids) {
        free(norm);
        free(pos);
        free(dp);
        free(prev);
        free(best_id);
        return -1;
    }

    int idx = end_idx;
    for (int i = count - 1; i >= 0; i--) {
        ids[i] = best_id[idx];
        idx = prev[idx];
    }

    *out_ids = ids;
    *out_len = count;

    free(norm);
    free(pos);
    free(dp);
    free(prev);
    free(best_id);
    return 0;
}
