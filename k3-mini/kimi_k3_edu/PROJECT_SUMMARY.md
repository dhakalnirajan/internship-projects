# K3-Edu: Complete Project Summary

- **Project:** K3-Edu ~200M Parameter Transformer
- **Inspiration:** MoonshotAI Kimi K3 Architecture
- **Language:** C (CPU) + CUDA (GPU)
- **Status:** Complete, Education-Ready Codebase

## Architecture Specifications

| Spec | Value |
|---|---|
| Model size | ~200M parameters |
| Embedding dim | 768 |
| Hidden layers | 12 |
| Attention heads | 12 |
| Head dimension | 64 |
| FFN dimension | 3072 (4x embedding) |
| Context window | 8192 tokens |
| Vocabulary size | ~50,000 (BPE) |

Key innovations from Kimi K3:
- Kimi Delta Attention (KDA): Linear attention with channel-wise gating
- Attention Residuals (AttnRes): Selective depth-wise aggregation
- 3:1 KDA:MLA ratio per block
- RMSNorm + RoPE + SwiGLU FFN
- FP8/INT8 quantization support

## File Structure

### include/
- `config.h` — Hyperparameters, special tokens, hardware config
- `tokenizer.h` — BPE tokenizer API with instruction tokens
- `model.h` — Transformer architecture, KDA, AttnRes, KV-cache
- `train.h` — AdamW, LR scheduler, dataset API, training state
- `inference.h` — Generation config, sampling, chat format, evaluation
- `utils.h` — Math ops, quantization, hardware detection, logging

### src/
- `config.c` — Config init/load/save, hardware auto-detection
- `tokenizer.c` — BPE training, encode/decode, instruction formatting
- `model.c` — Full Transformer: KDA attention, FFN, AttnRes, forward
- `train_base.c` — Phase 1 training: AdamW, cosine+warm restarts, plateau detection
- `train_inst.c` — Phase 2 instruction fine-tuning
- `inference.c` — Temperature/top-k/top-p sampling, KV-cache, chat API
- `data_loader.c` — Dataset loading, per-epoch shuffling, batching
- `utils.c` — Matmul, softmax, norm, quantization, file I/O
- `train_main.c` — Unified training CLI entry point
- `inference_main.c` — Interactive generation CLI
- `evaluate_main.c` — Perplexity evaluation CLI
- `tokenizer_main.c` — Tokenizer training/testing CLI

### cuda/
- `attention.cu` — KDA chunkwise kernel, FlashAttention, RoPE, RMSNorm, SwiGLU
- `matmul.cu` — Tiled GEMM, batched GEMM, transpose GEMM
- `train_gpu.cu` — GPU loss, gradient, AdamW, clipping kernels

### scripts/
- `prepare_data.py` — HuggingFace dataset download, multilingual + code
- `evaluate.py` — Perplexity, code accuracy, overfitting detection

### Build & Config
- `Makefile` — Complete build system with CUDA auto-detection
- `README.md` — Quick start guide
- `DOCUMENTATION.md` — Deep technical documentation
- `config.json` — Example configuration

## Training Pipeline

### Phase 1: Base Pretraining (150K steps)
- Data: 2-5GB diverse text (40% NL, 40% code, 20% docs)
- LR: 1e-5 -> 1e-3 -> 5e-4 -> 1e-5 (cosine + 3 warm restarts)
- Batch: 4-8 (auto-detected), grad accum 1-8
- Optimizations: weight decay 0.1, grad clip 1.0, per-epoch shuffle
- Early stopping: plateau detection (5 consecutive plateaus)

### Phase 2: Instruction Fine-Tuning (30K steps)
- Data: Instruction datasets with special tokens
- LR: 1e-5 -> 1e-6
- Format: `<|im_start|>system/user/assistant|>...|>text|>`
- Masked loss: only compute on assistant responses

## Inference Features

- Sampling: Temperature, top-k (40), top-p (0.9), repetition penalty
- Formats: Raw generation, chat completion, code generation
- Speed: KV-cache for O(1) per-token generation
- Precision: FP8 activations, INT8 weights (4x memory reduction)
- Hardware: CPU (OpenMP) and GPU (CUDA) execution

## Evaluation

- Perplexity, cross-entropy loss, code keyword accuracy
- Code execution accuracy (compile & run generated code)
- Overfitting detection (train/val gap analysis)
- Loss plateau detection with recommendations

## Hardware Adaptation

- Auto-detects: CPU cores, system RAM, GPU count, GPU memory
- Scales: Batch size 1-8, gradient accumulation 1-8
- Target: Effective batch of 32K tokens regardless of hardware

## Languages Supported

- Natural: English, Nepali
- Programming: Python, C, C++, Java, JavaScript, Rust, Go

## Special Tokens

`<|pad|>` (0), `<|unk|>` (1), `<|bos|>` (2), `<|eos|>` (3), `<|mask|>` (4), `<|im_start|>` (5), `|>` (6), `<|system|>` (7), `<|user|>` (8), `<|assistant|>` (9)

## Build Instructions

```bash
make all              # Build everything (CPU + GPU if CUDA available)
make build/train_base # CPU training only
make build/train_gpu  # GPU training (requires CUDA)
make test             # Verify build
```

## Usage Workflow

```bash
python scripts/prepare_data.py --output datasets/ --size 5GB
./build/train_tokenizer --dataset datasets/train/ --output tokenizer.bin
./build/train_base --dataset datasets/train/ --tokenizer tokenizer.bin
./build/train_inst --base-model checkpoints/best_base.bin --dataset datasets/train/
./build/inference --model checkpoints/best_inst.bin --tokenizer tokenizer.bin --prompt "Hello"
python scripts/evaluate.py --model checkpoints/best_inst.bin --tokenizer tokenizer.bin
```

## Educational Value

This codebase teaches:
- Transformer architecture from scratch (no frameworks)
- Linear attention mechanisms (KDA)
- Advanced residual connections (AttnRes)
- Quantization (FP8/INT8) for efficient inference
- GPU kernel programming (CUDA)
- Training optimization techniques (grad accum, warm restarts, etc.)
- Tokenization (BPE with special tokens)
- Dataset pipeline design
- Hardware-aware system design
