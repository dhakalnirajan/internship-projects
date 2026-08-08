/**
 * inference_main.c - CLI entry point for text generation
 */

#include "inference.h"
#include "model.h"
#include "tokenizer.h"
#include "config.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    config_init();
    config_auto_detect_hardware();

    char* model_path = NULL;
    char* tokenizer_path = NULL;
    char* prompt = "Hello, world!";
    int chat_mode = 0;
    char* system_msg = "You are a helpful assistant.";
    float temperature = 0.7f;
    int max_tokens = 512;
    int top_k = 40;
    float top_p = 0.9f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--chat") == 0) chat_mode = 1;
        else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) system_msg = argv[++i];
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = atof(argv[++i]);
    }

    if (!model_path || !tokenizer_path) {
        printf("Usage: %s --model <path> --tokenizer <path> [options]\n", argv[0]);
        printf("Options:\n");
        printf("  --prompt <text>      Input prompt\n");
        printf("  --chat               Enable chat mode\n");
        printf("  --system <text>      System message for chat\n");
        printf("  --temperature <f>    Sampling temperature (default: 0.7)\n");
        printf("  --max-tokens <n>     Max tokens to generate (default: 512)\n");
        printf("  --top-k <n>          Top-k filtering (default: 40)\n");
        printf("  --top-p <f>          Top-p filtering (default: 0.9)\n");
        return 1;
    }

    printf("Loading tokenizer from %s...\n", tokenizer_path);
    Tokenizer* tok = tokenizer_load(tokenizer_path);
    if (!tok) {
        printf("Error: Failed to load tokenizer\n");
        return 1;
    }

    printf("Loading model from %s...\n", model_path);
    TransformerModel* model = model_load(model_path, tok);
    if (!model) {
        printf("Error: Failed to load model\n");
        tokenizer_free(tok);
        return 1;
    }

    printf("Model loaded: %zu parameters\n", model->n_params);

    GenerationConfig cfg = DEFAULT_GENERATION_CONFIG;
    cfg.temperature = temperature;
    cfg.max_new_tokens = max_tokens;
    cfg.top_k = top_k;
    cfg.top_p = top_p;

    char output[65536];
    int n_generated;

    if (chat_mode) {
        printf("\n=== Chat Mode ===\n");
        printf("System: %s\n\n", system_msg);

        ChatMessage messages[2];
        messages[0].role = "system";
        messages[0].content = system_msg;
        messages[1].role = "user";
        messages[1].content = prompt;

        char* response = chat_completion(model, tok, messages, 2, &cfg);
        printf("Assistant: %s\n", response);
        free(response);
        n_generated = 0;
    } else {
        printf("\nPrompt: %s\n", prompt);
        printf("Generating...\n\n");

        n_generated = generate(model, tok, prompt, &cfg, output, sizeof(output));
        printf("Generated (%d tokens): %s\n", n_generated, output);
    }

    model_free(model);
    tokenizer_free(tok);

    return 0;
}
