cmake_minimum_required(VERSION 3.20)
project(K3Edu C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization and warning flags
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        add_compile_options(${OpenMP_C_FLAGS})
    endif()
endif()

# Check for CUDA support option
option(USE_CUDA "Enable CUDA support" ON)
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        add_definitions(-DUSE_CUDA)
        set(CUDA_FOUND TRUE)
    endif()
endif()

# Directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Ensure required directories exist at build generation time
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/checkpoints)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/datasets)

# Automatically generate missing entry points if needed
set(INFERENCE_MAIN_SRC ${CMAKE_SOURCE_DIR}/src/inference_main.c)
if(NOT EXISTS ${INFERENCE_MAIN_SRC})
    file(WRITE ${INFERENCE_MAIN_SRC}
"__FILE__LINE__
#include \"inference.h\"
#include \"model.h\"
#include \"tokenizer.h\"
#include \"config.h\"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    config_init();
    config_auto_detect_hardware();
    config_print();
    
    if (argc < 3) {
        printf(\"Usage: %s --model <path> --tokenizer <path> --prompt <text>\\n\", argv[0]);
        return 1;
    }
    
    char* model_path = NULL;
    char* tokenizer_path = NULL;
    char* prompt = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], \"--model\") == 0 && i + 1 < argc) model_path = argv[i+1];
        if (strcmp(argv[i], \"--tokenizer\") == 0 && i + 1 < argc) tokenizer_path = argv[i+1];
        if (strcmp(argv[i], \"--prompt\") == 0 && i + 1 < argc) prompt = argv[i+1];
    }
    
    if (!model_path || !tokenizer_path) {
        printf(\"Error: --model and --tokenizer required\\n\");
        return 1;
    }
    
    Tokenizer* tok = tokenizer_load(tokenizer_path);
    TransformerModel* model = model_load(model_path, tok);
    
    if (!prompt) prompt = \"Hello, world!\";
    
    GenerationConfig cfg = DEFAULT_GENERATION_CONFIG;
    char output[4096];
    int n = generate(model, tok, prompt, &cfg, output, sizeof(output));
    printf(\"Generated (%d tokens): %s\\n\", n, output);
    
    model_free(model);
    tokenizer_free(tok);
    return 0;
}
")
endif()

set(EVALUATE_MAIN_SRC ${CMAKE_SOURCE_DIR}/src/evaluate_main.c)
if(NOT EXISTS ${EVALUATE_MAIN_SRC})
    file(WRITE ${EVALUATE_MAIN_SRC}
"#include \"inference.h\"
#include \"model.h\"
#include \"tokenizer.h\"
#include \"config.h\"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 5) {
        printf(\"Usage: %s --model <path> --tokenizer <path> --text <text>\\n\", argv[0]);
        return 1;
    }
    
    char* model_path = NULL;
    char* tokenizer_path = NULL;
    char* text = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], \"--model\") == 0 && i + 1 < argc) model_path = argv[i+1];
        if (strcmp(argv[i], \"--tokenizer\") == 0 && i + 1 < argc) tokenizer_path = argv[i+1];
        if (strcmp(argv[i], \"--text\") == 0 && i + 1 < argc) text = argv[i+1];
    }
    
    Tokenizer* tok = tokenizer_load(tokenizer_path);
    TransformerModel* model = model_load(model_path, tok);
    
    float ppl = evaluate_perplexity(model, tok, text);
    printf(\"%.6f\\n\", ppl);
    
    model_free(model);
    tokenizer_free(tok);
    return 0;
}
")
endif()

# Common C Source files
set(COMMON_C_SOURCES
    src/tokenizer.c
    src/model.c
    src/inference.c
    src/utils.c
    src/config.c
    src/data_loader.c
)

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Targets
add_executable(train_tokenizer src/tokenizer_main.c ${COMMON_C_SOURCES})
add_executable(train_base src/train_base.c ${COMMON_C_SOURCES})
add_executable(train_inst src/train_inst.c ${COMMON_C_SOURCES})
add_executable(inference src/inference_main.c ${COMMON_C_SOURCES})
add_executable(evaluate src/evaluate_main.c ${COMMON_C_SOURCES})

# Target Link Libraries
foreach(target train_tokenizer train_base train_inst inference evaluate)
    target_link_libraries(${target} PRIVATE m pthread)
    if(OpenMP_C_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
    endif()
endforeach()

# CUDA specific target setup
if(CUDA_FOUND)
    set(CUDA_SOURCES
        cuda/train_gpu.cu
        cuda/attention.cu
        cuda/matmul.cu
    )
    
    # Check if CUDA source files exist before adding train_gpu
    set(CUDA_FILES_EXIST TRUE)
    foreach(cuda_file ${CUDA_SOURCES})
        if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${cuda_file})
            set(CUDA_FILES_EXIST FALSE)
        endif()
    endforeach()

    if(CUDA_FILES_EXIST)
        add_executable(train_gpu ${CUDA_SOURCES} src/model.c src/tokenizer.c src/utils.c src/config.c src/inference.c)
        set_target_properties(train_gpu PROPERTIES CUDA_ARCHITECTURES "70")
        target_link_libraries(train_gpu PRIVATE cudart cublas m pthread)
    endif()
endif()