/* k3_tokenizer.c — Simple word-piece fallback tokenizer */
#include "k3_tokenizer.h"
#include <ctype.h>

Tokenizer *tokenizer_load(const char *path) {
    (void)path;
    Tokenizer *t = calloc(1, sizeof(Tokenizer));
    t->max_vocab = 50000;
    t->vocab = calloc(t->max_vocab, sizeof(char*));
    /* Special tokens */
    t->vocab[TOK_PAD] = strdup("<|pad|>");
    t->vocab[TOK_UNK] = strdup("<|unk|>");
    t->vocab[TOK_BOS] = strdup("<|bos|>");
    t->vocab[TOK_EOS] = strdup("<|eos|>");
    t->vocab[TOK_IM_START] = strdup("<|im_start|>");
    t->vocab[TOK_IM_END]   = strdup("<|im_end|>");
    t->vocab[TOK_SYSTEM]   = strdup("<|system|>");
    t->vocab[TOK_USER]     = strdup("<|user|>");
    t->vocab[TOK_ASSISTANT] = strdup("<|assistant|>");
    t->vocab_size = 10;
    /* Simple char-level fallback vocab */
    for (int c = 32; c < 127 && t->vocab_size < t->max_vocab; c++) {
        char buf[8] = {0};
        buf[0] = (char)c;
        t->vocab[t->vocab_size++] = strdup(buf);
    }
    /* Common words */
    const char *words[] = {"the","a","is","are","was","were","be","been",
        "being","have","has","had","do","does","did","will","would","could",
        "should","may","might","must","shall","can","need","dare","ought",
        "used","to","of","in","for","on","with","at","by","from","as","into",
        "through","during","before","after","above","below","between","under",
        "and","but","or","yet","so","if","because","although","though","while",
        "where","when","that","which","who","whom","whose","what","this","these",
        "those","I","you","he","she","it","we","they","me","him","her","us","them",
        "my","your","his","its","our","their","mine","yours","hers","ours","theirs",
        "hello","world","how","what","why","where","when","which","who","good",
        "bad","new","old","first","last","long","great","little","own","other",
        "right","left","big","high","different","small","large","next","early",
        "young","important","few","public","same","able","quantum","computer",
        "computing","algorithm","data","model","network","learning","machine",
        "artificial","intelligence","neural","deep","training","inference",
        "token","attention","transformer","layer","weight","bias","gradient",
        "optimization","loss","function","matrix","vector","tensor","scalar",
        "dimension","space","time","energy","matter","particle","wave","field",
        "electron","photon","atom","molecule","cell","organism","life","earth",
        "universe","galaxy","star","planet","sun","moon","light","sound","heat",
        "cold","water","air","fire","metal","wood","stone","sand","glass","paper",
        "book","page","word","sentence","paragraph","story","novel","poem",
        "music","song","art","painting","sculpture","dance","theater","film",
        "movie","game","sport","play","run","walk","jump","swim","fly","drive",
        "ride","eat","drink","sleep","wake","dream","think","know","believe",
        "remember","forget","understand","learn","teach","study","work","job",
        "career","business","company","market","money","price","cost","value",
        "trade","buy","sell","pay","spend","save","invest","profit","loss",
        "risk","safe","danger","protect","defend","attack","fight","war",
        "peace","love","hate","joy","sad","anger","fear","hope","dream",
        "goal","plan","strategy","tactic","move","step","process","system",
        "structure","organization","order","chaos","pattern","design",
        "style","form","shape","color","red","green","blue","yellow",
        "black","white","gray","brown","pink","purple","orange","gold",
        "silver","bronze","iron","steel","copper","lead","tin","zinc",
        "carbon","oxygen","hydrogen","nitrogen","helium","neon","argon",
        "sodium","chlorine","calcium","potassium","magnesium","aluminum",
        "silicon","phosphorus","sulfur","fluorine","lithium","boron",
        "nitrogen","oxygen","fluorine","neon","sodium","magnesium",
        "aluminum","silicon","phosphorus","sulfur","chlorine","argon",
        NULL};
    for (int i = 0; words[i] && t->vocab_size < t->max_vocab; i++)
        t->vocab[t->vocab_size++] = strdup(words[i]);
    return t;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t);
}

static int find_token(Tokenizer *t, const char *s, int len) {
    for (int i = t->vocab_size - 1; i >= 0; i--) {
        if ((int)strlen(t->vocab[i]) == len && strncmp(t->vocab[i], s, len) == 0)
            return i;
    }
    return TOK_UNK;
}

int tokenizer_encode(Tokenizer *t, const char *text, int *ids, int max_len) {
    int n = 0;
    const char *p = text;
    while (*p && n < max_len) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        /* Try longest match */
        int best_len = 1, best_id = TOK_UNK;
        int max_try = k3_min(32, (int)strlen(p));
        for (int len = max_try; len >= 1; len--) {
            int id = find_token(t, p, len);
            if (id != TOK_UNK) { best_len = len; best_id = id; break; }
        }
        ids[n++] = best_id;
        p += best_len;
    }
    return n;
}

int tokenizer_decode(Tokenizer *t, const int *ids, int n_ids, char *text, int max_len) {
    int pos = 0;
    for (int i = 0; i < n_ids && pos < max_len - 1; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) id = TOK_UNK;
        const char *tok = t->vocab[id];
        int len = (int)strlen(tok);
        if (pos + len >= max_len - 1) break;
        /* Skip special tokens in output */
        if (id >= TOK_PAD && id <= TOK_ASSISTANT) continue;
        memcpy(text + pos, tok, len);
        pos += len;
        text[pos] = ' ';
        pos++;
    }
    if (pos > 0 && text[pos-1] == ' ') pos--;
    text[pos] = '\0';
    return pos;
}

const char *tokenizer_get_token(Tokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size) return "<?>";
    return t->vocab[id];
}
