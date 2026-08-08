/**
 * tokenizer.h - Byte Pair Encoding (BPE) Tokenizer
 * 
 * Inspired by Kimi K3's tokenizer approach:
 * - GPT-2 style pretokenization for code and natural language
 * - Special tokens for instruction tuning (<|im_start|>, <|im_end|>, etc.)
 * - Support for multilingual text (English, Nepali) and programming languages
 * - Pre-tokenizer regex handles code identifiers, whitespace, and unicode
 * 
 * The tokenizer is trained on the dataset directory and produces a vocabulary
 * that understands both natural language and code tokens.
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "config.h"
#include <stddef.h>

/* ============================================================================
 * SPECIAL TOKEN IDs
 * ============================================================================ */

enum SpecialTokenID {
    TOK_PAD = 0,
    TOK_UNK = 1,
    TOK_BOS = 2,
    TOK_EOS = 3,
    TOK_MASK = 4,
    TOK_IM_START = 5,
    TOK_IM_END = 6,
    TOK_SYSTEM = 7,
    TOK_USER = 8,
    TOK_ASSISTANT = 9,
    NUM_SPECIAL_TOKENS = 10
};

/* ============================================================================
 * TOKEN STRUCTURE
 * ============================================================================ */

typedef struct {
    char* text;         /* UTF-8 token string */
    int id;             /* Token ID */
    int freq;           /* Frequency in training data */
} Token;

/* ============================================================================
 * MERGE RULE
 * ============================================================================ */

typedef struct {
    int left;           /* Left token ID */
    int right;          /* Right token ID */
    int result;         /* Merged token ID */
    int rank;           /* Merge priority (lower = earlier) */
} MergeRule;

/* ============================================================================
 * TOKENIZER STATE
 * ============================================================================ */

typedef struct {
    /* Vocabulary */
    Token* vocab;
    int vocab_size;
    int vocab_capacity;

    /* Merge rules (sorted by rank) */
    MergeRule* merges;
    int n_merges;
    int merges_capacity;

    /* Fast lookup: pair -> merge rank (for encoding) */
    int* pair_to_rank;  /* 2D array: [vocab_size][vocab_size] */

    /* Special token strings */
    char* special_tokens[NUM_SPECIAL_TOKENS];

    /* Pre-tokenization regex pattern (simplified) */
    /* GPT-2 style: splits on whitespace, punctuation, numbers, etc. */

    /* Stats */
    size_t total_tokens_trained;
    size_t total_bytes_trained;
} Tokenizer;

/* ============================================================================
 * TOKENIZER API
 * ============================================================================ */

/* Create a new tokenizer */
Tokenizer* tokenizer_create(int initial_vocab_size);

/* Destroy tokenizer and free memory */
void tokenizer_free(Tokenizer* tok);

/* Train tokenizer on dataset directory */
int tokenizer_train(Tokenizer* tok, const char* dataset_dir, int target_vocab_size);

/* Save tokenizer to file (vocab + merges) */
int tokenizer_save(Tokenizer* tok, const char* path);

/* Load tokenizer from file */
Tokenizer* tokenizer_load(const char* path);

/* Encode text to token IDs */
int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len);

/* Decode token IDs to text */
char* tokenizer_decode(Tokenizer* tok, const int* ids, int len);

/* Encode a single token (for inference) */
int tokenizer_encode_single(Tokenizer* tok, const char* text);

/* Get token string by ID */
const char* tokenizer_get_token(Tokenizer* tok, int id);

/* Get ID by token string (or TOK_UNK if not found) */
int tokenizer_get_id(Tokenizer* tok, const char* token);

/* Print tokenizer statistics */
void tokenizer_print_stats(Tokenizer* tok);

/* ============================================================================
 * INSTRUCTION FORMATTING
 * ============================================================================ */

/* Format a conversation turn for instruction tuning */
char* format_instruction(const char* system_msg, const char* user_msg, const char* assistant_msg);

/* Format for base training (no special instruction tokens) */
char* format_base(const char* text);

/* Check if a token ID is a special token */
static inline int is_special_token(int id) {
    return id >= 0 && id < NUM_SPECIAL_TOKENS;
}

#endif /* TOKENIZER_H */
