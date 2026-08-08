/**
 * tokenizer_main.c - CLI for training and testing the tokenizer
 */

#include "tokenizer.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    char* dataset_dir = "datasets";
    char* output_path = "tokenizer.bin";
    char* test_text = NULL;
    int vocab_size = TOKENIZER_VOCAB_SIZE;
    int mode = 0; /* 0=train, 1=test, 2=info */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) dataset_dir = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vocab_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) { test_text = argv[++i]; mode = 1; }
        else if (strcmp(argv[i], "--info") == 0) mode = 2;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("K3-Edu Tokenizer Tool\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("\nCommands:\n");
            printf("  (no mode)           Train tokenizer on dataset\n");
            printf("  --test <text>       Test encode/decode\n");
            printf("  --info              Show tokenizer statistics\n");
            printf("\nOptions:\n");
            printf("  --dataset <dir>     Dataset directory (default: datasets)\n");
            printf("  --output <file>     Output file (default: tokenizer.bin)\n");
            printf("  --vocab-size <n>    Target vocabulary size (default: %d)\n", TOKENIZER_VOCAB_SIZE);
            return 0;
        }
    }

    if (mode == 0) {
        /* Train mode */
        printf("Training tokenizer...\n");
        printf("  Dataset: %s\n", dataset_dir);
        printf("  Target vocab size: %d\n", vocab_size);

        Tokenizer* tok = tokenizer_create(vocab_size);

        if (tokenizer_train(tok, dataset_dir, vocab_size) == 0) {
            tokenizer_save(tok, output_path);
            printf("Tokenizer saved to %s\n", output_path);
        } else {
            printf("Training failed.\n");
        }

        tokenizer_free(tok);

    } else if (mode == 1) {
        /* Test mode */
        printf("Loading tokenizer from %s...\n", output_path);
        Tokenizer* tok = tokenizer_load(output_path);
        if (!tok) {
            printf("Failed to load tokenizer.\n");
            return 1;
        }

        printf("Input: %s\n", test_text);

        int len;
        int* ids = tokenizer_encode(tok, test_text, &len);
        printf("Encoded (%d tokens): ", len);
        for (int i = 0; i < len && i < 20; i++) {
            printf("%d ", ids[i]);
        }
        if (len > 20) printf("...");
        printf("\n");

        char* decoded = tokenizer_decode(tok, ids, len);
        printf("Decoded: %s\n", decoded);

        free(ids);
        free(decoded);
        tokenizer_free(tok);

    } else if (mode == 2) {
        /* Info mode */
        printf("Loading tokenizer from %s...\n", output_path);
        Tokenizer* tok = tokenizer_load(output_path);
        if (!tok) {
            printf("Failed to load tokenizer.\n");
            return 1;
        }

        tokenizer_print_stats(tok);
        tokenizer_free(tok);
    }

    return 0;
}
