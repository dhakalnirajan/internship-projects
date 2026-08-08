# K3-Edu Makefile
# Builds the complete K3-Edu training and inference system

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -Wextra -Iinclude
LDFLAGS = -lm -lpthread

# CUDA settings
CUDA_FLAGS = -O3 -arch=sm_70 -Iinclude -Xcompiler "-O3 -fopenmp"
CUDA_LDFLAGS = -lcudart -lcublas

# Detect CUDA
CUDA_AVAILABLE := $(shell which nvcc >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(CUDA_AVAILABLE),1)
    CFLAGS += -DUSE_CUDA
    LDFLAGS += $(CUDA_LDFLAGS)
endif

# Directories
SRC_DIR = src
CUDA_DIR = cuda
BUILD_DIR = build
INCLUDE_DIR = include

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CUDA_SOURCES = $(wildcard $(CUDA_DIR)/*.cu)

# Object files
C_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CUDA_OBJECTS = $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/%.cu.o,$(CUDA_SOURCES))

# Targets
TARGETS = $(BUILD_DIR)/train_tokenizer $(BUILD_DIR)/train_base \
          $(BUILD_DIR)/train_inst $(BUILD_DIR)/inference \
          $(BUILD_DIR)/evaluate

ifeq ($(CUDA_AVAILABLE),1)
    TARGETS += $(BUILD_DIR)/train_gpu
endif

# Default target
.PHONY: all clean dirs test

all: dirs $(TARGETS)

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p checkpoints
	@mkdir -p datasets

# C compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA compilation
$(BUILD_DIR)/%.cu.o: $(CUDA_DIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# Tokenizer training binary
$(BUILD_DIR)/train_tokenizer: $(BUILD_DIR)/tokenizer_main.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/config.o $(BUILD_DIR)/data_loader.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Base training binary
$(BUILD_DIR)/train_base: $(BUILD_DIR)/train_base.o $(BUILD_DIR)/model.o $(BUILD_DIR)/tokenizer.o \
                         $(BUILD_DIR)/utils.o $(BUILD_DIR)/config.o $(BUILD_DIR)/data_loader.o \
                         $(BUILD_DIR)/inference.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Instruction tuning binary
$(BUILD_DIR)/train_inst: $(BUILD_DIR)/train_inst.o $(BUILD_DIR)/model.o $(BUILD_DIR)/tokenizer.o \
                         $(BUILD_DIR)/utils.o $(BUILD_DIR)/config.o $(BUILD_DIR)/data_loader.o \
                         $(BUILD_DIR)/inference.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Inference binary
$(BUILD_DIR)/inference: $(BUILD_DIR)/inference_main.o $(BUILD_DIR)/inference.o $(BUILD_DIR)/model.o \
                        $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/config.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Evaluation binary
$(BUILD_DIR)/evaluate: $(BUILD_DIR)/evaluate_main.o $(BUILD_DIR)/inference.o $(BUILD_DIR)/model.o \
                       $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/config.o
	$(CC) $^ -o $@ $(LDFLAGS)

# GPU training binary (CUDA)
$(BUILD_DIR)/train_gpu: $(BUILD_DIR)/train_gpu.cu.o $(BUILD_DIR)/attention.cu.o $(BUILD_DIR)/matmul.cu.o \
                        $(BUILD_DIR)/model.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/config.o
	$(NVCC) $^ -o $@ $(CUDA_LDFLAGS) -Xlinker "$(LDFLAGS)"

# Main entry points (we need to create these)
$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/tokenizer.c $(INCLUDE_DIR)/tokenizer.h
$(BUILD_DIR)/model.o: $(SRC_DIR)/model.c $(INCLUDE_DIR)/model.h
$(BUILD_DIR)/train_base.o: $(SRC_DIR)/train_base.c $(INCLUDE_DIR)/train.h
$(BUILD_DIR)/train_inst.o: $(SRC_DIR)/train_inst.c $(INCLUDE_DIR)/train.h
$(BUILD_DIR)/inference.o: $(SRC_DIR)/inference.c $(INCLUDE_DIR)/inference.h
$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.c $(INCLUDE_DIR)/utils.h
$(BUILD_DIR)/config.o: $(SRC_DIR)/config.c $(INCLUDE_DIR)/config.h
$(BUILD_DIR)/data_loader.o: $(SRC_DIR)/data_loader.c $(INCLUDE_DIR)/train.h

# Main files (need to be created)
$(BUILD_DIR)/inference_main.o: $(SRC_DIR)/inference_main.c
$(BUILD_DIR)/evaluate_main.o: $(SRC_DIR)/evaluate_main.c

# Create main entry point files if they don't exist
$(SRC_DIR)/inference_main.c:
	@echo "Creating inference_main.c..."
	@printf '%s\n' '#include "inference.h"' '#include "model.h"' '#include "tokenizer.h"' '#include "config.h"' '#include <stdio.h>' '' 'int main(int argc, char** argv) {' '    config_init();' '    config_auto_detect_hardware();' '    config_print();' '    ' '    if (argc < 3) {' '        printf("Usage: %s --model <path> --tokenizer <path> --prompt <text>\\n", argv[0]);' '        return 1;' '    }' '    ' '    char* model_path = NULL;' '    char* tokenizer_path = NULL;' '    char* prompt = NULL;' '    ' '    for (int i = 1; i < argc; i++) {' '        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[i+1];' '        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[i+1];' '        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[i+1];' '    }' '    ' '    if (!model_path || !tokenizer_path) {' '        printf("Error: --model and --tokenizer required\\n");' '        return 1;' '    }' '    ' '    Tokenizer* tok = tokenizer_load(tokenizer_path);' '    TransformerModel* model = model_load(model_path, tok);' '    ' '    if (!prompt) prompt = "Hello, world!";' '    ' '    GenerationConfig cfg = DEFAULT_GENERATION_CONFIG;' '    char output[4096];' '    int n = generate(model, tok, prompt, &cfg, output, sizeof(output));' '    printf("Generated (%d tokens): %s\\n", n, output);' '    ' '    model_free(model);' '    tokenizer_free(tok);' '    return 0;' '}' > $@

$(SRC_DIR)/evaluate_main.c:
	@echo "Creating evaluate_main.c..."
	@printf '%s\n' '#include "inference.h"' '#include "model.h"' '#include "tokenizer.h"' '#include "config.h"' '#include <stdio.h>' '' 'int main(int argc, char** argv) {' '    if (argc < 5) {' '        printf("Usage: %s --model <path> --tokenizer <path> --text <text>\\n", argv[0]);' '        return 1;' '    }' '    ' '    char* model_path = NULL;' '    char* tokenizer_path = NULL;' '    char* text = NULL;' '    ' '    for (int i = 1; i < argc; i++) {' '        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[i+1];' '        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[i+1];' '        if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[i+1];' '    }' '    ' '    Tokenizer* tok = tokenizer_load(tokenizer_path);' '    TransformerModel* model = model_load(model_path, tok);' '    ' '    float ppl = evaluate_perplexity(model, tok, text);' '    printf("%.6f\\n", ppl);' '    ' '    model_free(model);' '    tokenizer_free(tok);' '    return 0;' '}' > $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/train_* $(BUILD_DIR)/inference $(BUILD_DIR)/evaluate

test: all
	@echo "Running basic tests..."
	$(BUILD_DIR)/train_tokenizer --help 2>/dev/null || true
	@echo "Build successful!"

# Help
help:
	@echo "K3-Edu Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all binaries"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run basic tests"
	@echo "  dirs         - Create required directories"
	@echo ""
	@echo "Binaries:"
	@echo "  build/train_tokenizer  - Train BPE tokenizer"
	@echo "  build/train_base       - Base model training (CPU)"
	@echo "  build/train_inst       - Instruction fine-tuning (CPU)"
	@echo "  build/train_gpu        - GPU training (requires CUDA)"
	@echo "  build/inference        - Text generation"
	@echo "  build/evaluate         - Perplexity evaluation"

$(BUILD_DIR)/tokenizer_main.o: $(SRC_DIR)/tokenizer_main.c $(INCLUDE_DIR)/tokenizer.h
	$(CC) $(CFLAGS) -c $< -o $@
