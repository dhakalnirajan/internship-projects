# K3-Edu: Educational Transformer (~200M Parameters)

A complete, education-ready implementation of a dense Transformer language model inspired by MoonshotAI's Kimi K3 architecture. Built from scratch in C with optional CUDA acceleration.

## Architecture Overview

| Component | Implementation | Description |
|-----------|---------------|-------------|
| **Attention** | Kimi Delta Attention (KDA) + MLA | Linear attention with channel-wise gating; 3:1 KDA:MLA ratio |
| **Residuals** | Attention Residuals (AttnRes) | Selective depth-wise aggregation |
| **Norm** | RMSNorm | Root-mean-square normalization |
| **Position** | RoPE | Rotary Position Embeddings |
| **FFN** | SwiGLU | Gated Linear Unit with Swish activation |
| **Precision** | FP8/INT8 | Quantization-aware training and inference |

### Model Specifications (~200M Parameters)

- d_model: 768
- n_layers: 12
- n_heads: 12
- head_dim: 64
- d_ffn: 3072
- max_seq_len: 8192
- vocab_size: ~50,000

## Quick Start

### 1. Prerequisites

```bash
sudo apt-get install build-essential libomp-dev
pip install datasets transformers numpy
```

### 2. Build

```bash
cd kimi_k3_edu
make all
```

### 3. Prepare Dataset

```bash
python scripts/prepare_data.py --output datasets/ --size 5GB
```

### 4. Train

```bash
# Train tokenizer
./build/train_tokenizer --dataset datasets/train/ --output tokenizer.bin

# Base training
./build/train_base --dataset datasets/train/ --tokenizer tokenizer.bin

# Instruction tuning
./build/train_inst --base-model checkpoints/best_base.bin --dataset datasets/train/
```

### 5. Inference

```bash
./build/inference --model checkpoints/best_inst.bin --tokenizer tokenizer.bin \
    --prompt "Write a Python function to sort a list:"
```

### 6. Evaluate

```bash
python scripts/evaluate.py --model checkpoints/best_inst.bin \
    --tokenizer tokenizer.bin --dataset datasets/val/
```

## Training Optimizations

- Gradient accumulation for effective larger batches
- Weight decay (0.1) excluding biases and LayerNorm
- Per-epoch shuffling with deterministic seeds
- Validation loss tracking for early stopping
- Gradient clipping at 1.0
- Dynamic batch sizing based on hardware
- Cosine decay with warm restarts
- Loss plateau detection
- FP8/INT8 quantization
- KV cache for efficient inference

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<|pad|>` | 0 | Padding |
| `<|unk|>` | 1 | Unknown token |
| `<|bos|>` | 2 | Beginning of sequence |
| `<|eos|>` | 3 | End of sequence |
| `<|mask|>` | 4 | Masked token |
| `<|im_start|>` | 5 | Start of message |
| `<|im_end|>` | 6 | End of message |
| `<|system|>` | 7 | System role |
| `<|user|>` | 8 | User role |
| `<|assistant|>` | 9 | Assistant role |

## Project Structure

```
kimi_k3_edu/
├── include/          # Header files
├── src/              # C implementations
├── cuda/             # GPU kernels
├── scripts/          # Python utilities
├── datasets/         # Training data
└── checkpoints/      # Saved models
```

## License

MIT License - Educational use only.

## References

- Kimi K3 Technical Blog: https://kimi.com/blog/kimi-k3
- Kimi Linear (arXiv:2510.26692)
- Attention Residuals (arXiv:2603.15031)
