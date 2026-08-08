# K3-Edu Technical Documentation

## Table of Contents

1. [Architecture Deep Dive](#architecture-deep-dive)
2. [Training Pipeline](#training-pipeline)
3. [Inference Engine](#inference-engine)
4. [CUDA Kernels](#cuda-kernels)
5. [Dataset Format](#dataset-format)
6. [Hyperparameter Guide](#hyperparameter-guide)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Deep Dive

### Kimi Delta Attention (KDA)

KDA replaces standard softmax attention with a linear attention mechanism using a delta rule update:

```
S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1} + beta_t * k_t * v_t^T
```

Where:

- `S_t`: Memory state matrix [head_dim x head_dim]
- `alpha_t`: Per-channel forget gate (vector)
- `beta_t`: Learning rate-like update strength (scalar)
- `k_t, v_t`: Key and value vectors at position t

**Benefits:**

- O(1) memory per token (vs O(seq_len) for standard attention)
- Chunkwise parallelization for GPU efficiency
- No softmax bottleneck

**Implementation:** `src/model.c` - `kda_attention_forward()`

### Attention Residuals (AttnRes)

Standard Transformers use fixed residual connections: `h_l = h_{l-1} + Attention(h_{l-1})`

AttnRes replaces this with a learned, depth-wise attention mechanism:

```
h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
where alpha = softmax(w_l^T * RMSNorm(v_i))
```

**Block AttnRes** (used in K3-Edu):

- Divide 12 layers into 4 blocks of 3 layers each
- Standard residuals within blocks
- Between blocks, each layer attends to all previous block summaries

**Benefits:**

- Selective information flow (not all layers carry equal weight)
- Mitigates over-smoothing in deep networks
- Improves training stability

**Implementation:** `src/model.c` - `attnres_forward()`

### SwiGLU Feed-Forward

```
FFN(x) = (x * W_gate) * silu(x * W_up) * W_down
```

Where `silu(x) = x * sigmoid(x)`

**Benefits over ReLU/GeLU:**

- Gating mechanism filters irrelevant information
- Better gradient flow
- Standard in modern LLMs (PaLM, LLaMA, etc.)

**Implementation:** `src/model.c` - `ffn_forward()`

---

## Training Pipeline

### Phase 1: Base Pretraining

**Objective:** Learn general language and code understanding

**Data:**

- 2-5GB of diverse, high-quality text
- Mix: 40% natural language, 40% code, 20% technical docs
- Languages: English, Nepali, Python, C, C++, Java, JavaScript, Rust, Go

**Hyperparameters:**

```
Peak LR:        3e-4
Warmup:         2,000 steps
Total steps:    150,000
Batch size:     4-8 (auto-detected)
Grad accum:     1-8 (target: 32K tokens)
Weight decay:   0.1 (excl. biases/norms)
Dropout:        0.1
Grad clip:      1.0
```

**LR Schedule:**

```
Step 0-2K:      Linear warmup 1e-5 -> 1e-3
Step 2K-50K:    Cosine decay 1e-3 -> 5e-4
Step 50K-150K:  Cosine + warm restarts 5e-4 -> 1e-5
                (3 restarts, period doubles each time)
```

**Why 2-5GB is the sweet spot for 200M params:**

- Chinchilla scaling laws suggest ~4 tokens/param for optimal training
- 200M params * 4 = 800M tokens ≈ 2-4GB of text
- Beyond this, the model saturates and overfitting becomes severe
- Per-epoch shuffling prevents memorization

### Phase 2: Instruction Fine-Tuning

**Objective:** Learn to follow instructions and engage in conversation

**Data:**

- Instruction-style datasets (Alpaca, OpenOrca, CodeAlpaca)
- Formatted with special tokens

**Format:**

```
<|im_start|>system
You are a helpful assistant.}
<|im_start|>user
What is the capital of France?}
<|im_start|>assistant
The capital of France is Paris.}
```

**Hyperparameters:**

```
Peak LR:        1e-5
Min LR:         1e-6
Warmup:         500 steps
Total steps:    30,000 (20% of base training)
Batch size:     2
Weight decay:   0.01
Dropout:        0.05
```

**Key difference from base training:**

- Only compute loss on assistant responses
- Mask user/system tokens in loss computation
- Lower learning rate to preserve base knowledge

---

## Inference Engine

### Sampling Methods

**Temperature Sampling:**

```python
p_i = exp(logit_i / T) / sum(exp(logit_j / T))
```

- T -> 0: Greedy (deterministic)
- T = 1.0: True distribution
- T > 1.0: More random/creative

**Top-k Filtering:**

- Keep only k highest probability tokens
- Set others to -infinity
- Typical: k=40

**Top-p (Nucleus) Filtering:**

- Sort tokens by probability
- Keep smallest set where cumulative probability >= p
- Typical: p=0.9

**Combined:**

1. Apply temperature scaling
2. Apply top-k filter
3. Apply top-p filter
4. Sample from remaining distribution

### KV Cache

During autoregressive generation, keys and values from previous tokens can be reused:

```
Without cache: O(seq_len^2) per token
With cache:    O(seq_len) per token
```

**Implementation:**

- Allocate max_seq_len cache at startup
- Append new K,V after each forward pass
- Clear cache between conversations

---

## CUDA Kernels

### Attention Kernels

**KDA Forward:**

- One block per (batch, head, chunk)
- Shared memory for S matrix
- Sequential processing within chunk, parallel across chunks

**Flash Attention:**

- One thread per query position
- Online softmax to avoid materializing full attention matrix
- Causal masking built-in

### GEMM Kernels

**Tiled Matrix Multiply:**

- TILE_M=64, TILE_N=64, TILE_K=32
- Shared memory for A and B tiles
- Coalesced memory access patterns

**Batched GEMM:**

- For multi-head attention
- 3D grid: (N, M, batch)

---

## Dataset Format

### File Structure

```
datasets/
├── train/
│   ├── nl_0000.txt      # Natural language
│   ├── nl_0001.txt
│   ├── code_0000.txt    # Code
│   ├── code_0001.txt
│   └── instr_0000.txt   # Instructions
└── val/
    ├── nl_0000.txt
    └── code_0000.txt
```

### Text Format

Files contain multiple samples separated by:

```

===SAMPLE===

```

### Instruction Format

```
<|im_start|>system
You are a helpful coding assistant.}
<|im_start|>user
Write a Python function to calculate factorial.}
<|im_start|>assistant
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)}
```

---

## Hyperparameter Guide

### For Different Hardware

| Hardware | Batch | Grad Accum | LR | Notes |
| ---------- | ------- | ----------- | ----- | ------- |
| RTX 4090 (24GB) | 8 | 1 | 3e-4 | Full speed |
| RTX 3090 (24GB) | 8 | 1 | 3e-4 | Full speed |
| RTX 4080 (16GB) | 6 | 1 | 3e-4 | Slightly slower |
| RTX 4070 (12GB) | 4 | 2 | 3e-4 | Good efficiency |
| RTX 4060 (8GB) | 2 | 4 | 2e-4 | Reduce LR slightly |
| CPU (32GB RAM) | 4 | 2 | 3e-4 | Use all cores |
| CPU (16GB RAM) | 2 | 4 | 3e-4 | May be slow |
| CPU (8GB RAM) | 1 | 8 | 2e-4 | Reduce LR, expect long training |

### Overfitting Detection

**Signs of overfitting:**

1. Training loss decreases but validation loss increases
2. Validation loss plateaus for >20 steps
3. Model generates memorized sequences
4. Perplexity on validation >> training

**Solutions:**

1. Reduce training data size (counterintuitive but effective for small models)
2. Increase dropout (0.1 -> 0.15)
3. Increase weight decay (0.1 -> 0.15)
4. Add more diverse data
5. Use early stopping (plateau detection)
6. Reduce learning rate

---

## Troubleshooting

### Build Issues

**"cuda_runtime.h not found"**

- Install CUDA Toolkit or build without CUDA: `make build/train_base`

**"omp.h not found"**

- Install OpenMP: `sudo apt-get install libomp-dev`

**"undefined reference to expf"**

- Link math library: Add `-lm` to LDFLAGS (already in Makefile)

### Training Issues

**"NaN loss"**

- Reduce learning rate by 10x
- Check for gradient explosion (enable gradient clipping)
- Verify data doesn't contain invalid characters

**"Loss not decreasing"**

- Check learning rate (too high or too low)
- Verify data quality (random text = no learning)
- Ensure shuffling is working (check logs)
- Try warm restart (cosine schedule with restarts)

**"Out of memory"**

- Reduce batch size
- Increase gradient accumulation
- Reduce sequence length (8192 -> 4096)
- Enable FP8/INT8 quantization

### Inference Issues

**"Generated text is repetitive"**

- Increase temperature (0.7 -> 1.0)
- Enable repetition penalty (1.0 -> 1.2)
- Reduce top_p (0.9 -> 0.8)

**"Generated text is gibberish"**

- Model may be undertrained (need more steps)
- Check tokenizer matches training
- Verify model weights loaded correctly

**"Slow generation"**

- Enable KV cache (should be default)
- Use GPU inference if available
- Reduce max_seq_len if not needed
- Consider INT8 quantization for 2x speedup
