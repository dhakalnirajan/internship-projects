/* k3_tokenizer.h — BPE tokenizer */
#ifndef K3_TOKENIZER_H
#define K3_TOKENIZER_H
#include "k3_common.h"

#define TOK_PAD      0
#define TOK_UNK      1
#define TOK_BOS      2
#define TOK_EOS      3
#define TOK_IM_START 5
#define TOK_IM_END   6
#define TOK_SYSTEM   7
#define TOK_USER     8
#define TOK_ASSISTANT 9

typedef struct {
    char **vocab;
    int vocab_size;
    int max_vocab;
} Tokenizer;

Tokenizer *tokenizer_load(const char *path);
void tokenizer_free(Tokenizer *t);
int tokenizer_encode(Tokenizer *t, const char *text, int *ids, int max_len);
int tokenizer_decode(Tokenizer *t, const int *ids, int n_ids, char *text, int max_len);
const char *tokenizer_get_token(Tokenizer *t, int id);

#endif
