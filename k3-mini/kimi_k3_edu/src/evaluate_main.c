/**
 * evaluate_main.c - CLI entry point for perplexity evaluation
 */

#include "inference.h"
#include "model.h"
#include "tokenizer.h"
#include "config.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    char* model_path = NULL;
    char* tokenizer_path = NULL;
    char* text = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
    }

    if (!model_path || !tokenizer_path || !text) {
        printf("Usage: %s --model <path> --tokenizer <path> --text <text>\n", argv[0]);
        return 1;
    }

    Tokenizer* tok = tokenizer_load(tokenizer_path);
    if (!tok) {
        printf("Error: Failed to load tokenizer\n");
        return 1;
    }

    TransformerModel* model = model_load(model_path, tok);
    if (!model) {
        printf("Error: Failed to load model\n");
        tokenizer_free(tok);
        return 1;
    }

    float ppl = evaluate_perplexity(model, tok, text);
    printf("%.6f\n", ppl);

    model_free(model);
    tokenizer_free(tok);

    return 0;
}
