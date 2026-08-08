/**
 * data_loader.c - Dataset Loading with Shuffling
 * 
 * Reads .txt and .markdown files from datasets/ directory.
 * Supports per-epoch shuffling, dynamic batch sizing, and
 * efficient tokenization caching.
 */

#include "train.h"
#include "utils.h"
#include <dirent.h>
#include <sys/stat.h>

/* ============================================================================
 * FILE DISCOVERY
 * ============================================================================ */

typedef struct {
    char** paths;
    size_t* sizes;
    int count;
    int capacity;
} FileList;

static FileList* file_list_create(void) {
    FileList* list = (FileList*)calloc(1, sizeof(FileList));
    list->capacity = 256;
    list->paths = (char**)malloc(list->capacity * sizeof(char*));
    list->sizes = (size_t*)malloc(list->capacity * sizeof(size_t));
    return list;
}

static void file_list_free(FileList* list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) {
        free(list->paths[i]);
    }
    free(list->paths);
    free(list->sizes);
    free(list);
}

static void file_list_add(FileList* list, const char* path) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->paths = (char**)realloc(list->paths, list->capacity * sizeof(char*));
        list->sizes = (size_t*)realloc(list->sizes, list->capacity * sizeof(size_t));
    }

    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        list->paths[list->count] = strdup(path);
        list->sizes[list->count] = st.st_size;
        list->count++;
    }
}

static void scan_directory_recursive(const char* dir, FileList* list) {
    DIR* d = opendir(dir);
    if (!d) {
        log_error("Cannot open directory: %s", dir);
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);

        if (entry->d_type == DT_DIR) {
            scan_directory_recursive(path, list);
        } else if (is_text_file(path)) {
            file_list_add(list, path);
        }
    }

    closedir(d);
}

/* ============================================================================
 * TOKENIZED CHUNK
 * ============================================================================ */

typedef struct {
    int* tokens;
    size_t len;
    size_t capacity;
} TokenChunk;

static TokenChunk* chunk_create(void) {
    TokenChunk* chunk = (TokenChunk*)calloc(1, sizeof(TokenChunk));
    chunk->capacity = 1024 * 1024; /* 1M tokens initial */
    chunk->tokens = (int*)malloc(chunk->capacity * sizeof(int));
    return chunk;
}

static void chunk_free(TokenChunk* chunk) {
    if (!chunk) return;
    free(chunk->tokens);
    free(chunk);
}

static void chunk_append(TokenChunk* chunk, const int* tokens, size_t n) {
    if (chunk->len + n > chunk->capacity) {
        while (chunk->len + n > chunk->capacity) {
            chunk->capacity *= 2;
        }
        chunk->tokens = (int*)realloc(chunk->tokens, chunk->capacity * sizeof(int));
    }
    memcpy(chunk->tokens + chunk->len, tokens, n * sizeof(int));
    chunk->len += n;
}

/* ============================================================================
 * DATASET LOADER
 * ============================================================================ */

typedef struct {
    FileList* files;
    TokenChunk** chunks;
    int n_chunks;
    int chunk_capacity;

    /* Shuffling */
    int* shuffle_order;
    int shuffle_seed;
    int current_chunk;
    int epoch;

    /* Batching */
    int seq_len;
    int batch_size;
    size_t pos_in_chunk;

    /* Tokenizer reference */
    Tokenizer* tokenizer;
} DataLoader;

DataLoader* data_loader_create(const char* dir, Tokenizer* tokenizer, 
                                int seq_len, int batch_size) {
    DataLoader* loader = (DataLoader*)calloc(1, sizeof(DataLoader));
    loader->files = file_list_create();
    loader->tokenizer = tokenizer;
    loader->seq_len = seq_len;
    loader->batch_size = batch_size;
    loader->pos_in_chunk = 0;
    loader->current_chunk = 0;
    loader->epoch = 0;

    /* Scan directory */
    scan_directory_recursive(dir, loader->files);
    log_info("Found %d files in dataset", loader->files->count);

    if (loader->files->count == 0) {
        file_list_free(loader->files);
        free(loader);
        return NULL;
    }

    /* Pre-tokenize all files into chunks */
    loader->chunk_capacity = loader->files->count;
    loader->chunks = (TokenChunk**)calloc(loader->chunk_capacity, sizeof(TokenChunk*));
    loader->shuffle_order = (int*)malloc(loader->chunk_capacity * sizeof(int));

    log_info("Tokenizing dataset...");
    size_t total_tokens = 0;

    for (int i = 0; i < loader->files->count; i++) {
        size_t text_len;
        char* text = read_file(loader->files->paths[i], &text_len);
        if (!text) continue;

        int n_tokens;
        int* tokens = tokenizer_encode(tokenizer, text, &n_tokens);

        if (n_tokens > 0) {
            loader->chunks[loader->n_chunks] = chunk_create();
            chunk_append(loader->chunks[loader->n_chunks], tokens, n_tokens);
            loader->shuffle_order[loader->n_chunks] = loader->n_chunks;
            loader->n_chunks++;
            total_tokens += n_tokens;
        }

        free(tokens);
        free(text);
    }

    log_info("Dataset tokenized: %zu tokens in %d chunks", total_tokens, loader->n_chunks);

    /* Initial shuffle */
    data_loader_shuffle(loader, 42);

    return loader;
}

void data_loader_free(DataLoader* loader) {
    if (!loader) return;
    file_list_free(loader->files);
    for (int i = 0; i < loader->n_chunks; i++) {
        chunk_free(loader->chunks[i]);
    }
    free(loader->chunks);
    free(loader->shuffle_order);
    free(loader);
}

void data_loader_shuffle(DataLoader* loader, int seed) {
    loader->shuffle_seed = seed;
    srand(seed);

    /* Fisher-Yates shuffle on chunk order */
    for (int i = loader->n_chunks - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = loader->shuffle_order[i];
        loader->shuffle_order[i] = loader->shuffle_order[j];
        loader->shuffle_order[j] = tmp;
    }

    loader->current_chunk = 0;
    loader->pos_in_chunk = 0;
    loader->epoch++;

    log_info("DataLoader shuffled for epoch %d (seed=%d)", loader->epoch, seed);
}

int data_loader_next_batch(DataLoader* loader, int* input_ids, int* target_ids) {
    int filled = 0;

    while (filled < loader->batch_size) {
        if (loader->current_chunk >= loader->n_chunks) {
            break; /* Epoch complete */
        }

        int chunk_idx = loader->shuffle_order[loader->current_chunk];
        TokenChunk* chunk = loader->chunks[chunk_idx];

        while (filled < loader->batch_size && 
               loader->pos_in_chunk + loader->seq_len < chunk->len) {

            for (int s = 0; s < loader->seq_len; s++) {
                input_ids[filled * loader->seq_len + s] = 
                    chunk->tokens[loader->pos_in_chunk + s];
                target_ids[filled * loader->seq_len + s] = 
                    chunk->tokens[loader->pos_in_chunk + s + 1];
            }

            loader->pos_in_chunk += loader->seq_len;
            filled++;
        }

        if (loader->pos_in_chunk + loader->seq_len >= chunk->len) {
            /* Move to next chunk */
            loader->current_chunk++;
            loader->pos_in_chunk = 0;
        }
    }

    return filled;
}

int data_loader_is_epoch_done(DataLoader* loader) {
    return loader->current_chunk >= loader->n_chunks;
}

size_t data_loader_total_tokens(DataLoader* loader) {
    size_t total = 0;
    for (int i = 0; i < loader->n_chunks; i++) {
        total += loader->chunks[i]->len;
    }
    return total;
}

/* ============================================================================
 * DATASET API (compatibility with train.h)
 * ============================================================================ */

Dataset* dataset_create(const char* dir, int seq_len, int batch_size) {
    /* This is a simplified version - in practice, tokenizer is needed */
    /* The actual implementation creates a DataLoader internally */
    Dataset* ds = (Dataset*)calloc(1, sizeof(Dataset));
    ds->seq_len = seq_len;
    ds->batch_size = batch_size;
    ds->dataset_dir = strdup(dir);

    /* List files for compatibility */
    ds->file_paths = list_dataset_files(dir, &ds->n_files);
    ds->shuffle_indices = (int*)malloc(ds->n_files * sizeof(int));
    for (int i = 0; i < ds->n_files; i++) {
        ds->shuffle_indices[i] = i;
    }

    return ds;
}

void dataset_free(Dataset* ds) {
    if (!ds) return;
    for (int i = 0; i < ds->n_files; i++) free(ds->file_paths[i]);
    free(ds->file_paths);
    free(ds->token_buffer);
    free(ds->shuffle_indices);
    free(ds->dataset_dir);
    free(ds);
}

void dataset_shuffle(Dataset* ds, int seed) {
    ds->shuffle_seed = seed;
    srand(seed);
    for (int i = ds->n_files - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = ds->shuffle_indices[i];
        ds->shuffle_indices[i] = ds->shuffle_indices[j];
        ds->shuffle_indices[j] = tmp;
    }
    ds->epoch++;
    ds->current_file = 0;
}

void dataset_reset(Dataset* ds) {
    ds->current_file = 0;
    ds->buffer_pos = 0;
}

int dataset_next_batch(Dataset* ds, int* input_ids, int* target_ids,
                        int batch_size, int seq_len) {
    /* Simplified placeholder - real implementation uses DataLoader */
    if (ds->current_file >= ds->n_files) return 0;

    size_t len;
    char* text = read_file(ds->file_paths[ds->shuffle_indices[ds->current_file]], &len);
    if (!text) {
        ds->current_file++;
        return dataset_next_batch(ds, input_ids, target_ids, batch_size, seq_len);
    }

    /* Simple byte-level tokenization for placeholder */
    int filled = 0;
    for (int b = 0; b < batch_size && (filled + 1) * seq_len < (int)len; b++) {
        for (int s = 0; s < seq_len; s++) {
            input_ids[b * seq_len + s] = (unsigned char)text[filled * seq_len + s];
            target_ids[b * seq_len + s] = (unsigned char)text[filled * seq_len + s + 1];
        }
        filled++;
    }

    free(text);
    ds->current_file++;
    return filled;
}

size_t dataset_count_tokens(Dataset* ds, Tokenizer* tokenizer) {
    size_t total = 0;
    for (int i = 0; i < ds->n_files; i++) {
        size_t len;
        char* text = read_file(ds->file_paths[i], &len);
        if (text) {
            int n;
            int* tokens = tokenizer_encode(tokenizer, text, &n);
            total += n;
            free(tokens);
            free(text);
        }
    }
    return total;
}

void dataset_split(Dataset* ds, float val_ratio, Dataset** train_ds, Dataset** val_ds) {
    int val_count = (int)(ds->n_files * val_ratio);
    int train_count = ds->n_files - val_count;

    *train_ds = (Dataset*)calloc(1, sizeof(Dataset));
    *val_ds = (Dataset*)calloc(1, sizeof(Dataset));

    (*train_ds)->seq_len = ds->seq_len;
    (*train_ds)->batch_size = ds->batch_size;
    (*train_ds)->n_files = train_count;
    (*train_ds)->file_paths = (char**)malloc(train_count * sizeof(char*));
    (*train_ds)->shuffle_indices = (int*)malloc(train_count * sizeof(int));

    (*val_ds)->seq_len = ds->seq_len;
    (*val_ds)->batch_size = ds->batch_size;
    (*val_ds)->n_files = val_count;
    (*val_ds)->file_paths = (char**)malloc(val_count * sizeof(char*));
    (*val_ds)->shuffle_indices = (int*)malloc(val_count * sizeof(int));

    for (int i = 0; i < train_count; i++) {
        (*train_ds)->file_paths[i] = strdup(ds->file_paths[i]);
        (*train_ds)->shuffle_indices[i] = i;
    }

    for (int i = 0; i < val_count; i++) {
        (*val_ds)->file_paths[i] = strdup(ds->file_paths[train_count + i]);
        (*val_ds)->shuffle_indices[i] = i;
    }
}
