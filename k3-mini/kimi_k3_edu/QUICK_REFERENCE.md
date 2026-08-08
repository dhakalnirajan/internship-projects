# K3-Edu Quick Reference

## Build
```bash
make all                    # Build everything
make clean                  # Clean build artifacts
```

## Data Preparation
```bash
python scripts/prepare_data.py --output datasets/ --size 5GB
```

## Training
```bash
# Phase 1: Base
./build/train_base --dataset datasets/train/ --tokenizer tokenizer.bin

# Phase 2: Instruction
./build/train_inst --base-model checkpoints/best_base.bin --dataset datasets/train/
```

## Inference
```bash
# Single prompt
./build/inference --model checkpoints/best_inst.bin --tokenizer tokenizer.bin     --prompt "Write a Python function to sort a list:"

# Chat mode
./build/inference --model checkpoints/best_inst.bin --tokenizer tokenizer.bin     --chat --system "You are a coding assistant" --prompt "How do I use recursion?"
```

## Evaluation
```bash
python scripts/evaluate.py --model checkpoints/best_inst.bin     --tokenizer tokenizer.bin --dataset datasets/val/
```

## Key Files
- `config.json` - Edit hyperparameters
- `training.log` - Monitor training progress
- `checkpoints/` - Saved model weights
- `datasets/` - Training data (add .txt/.md files)

## Hyperparameters (config.json)
| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 768 | Embedding dimension |
| n_layers | 12 | Transformer layers |
| n_heads | 12 | Attention heads |
| batch_size | 4 | Per-device batch |
| peak_lr | 3e-4 | Maximum learning rate |
| warmup_steps | 2000 | Linear warmup |
| total_steps | 150000 | Training steps |
| dropout | 0.1 | Regularization |
| weight_decay | 0.1 | L2 penalty |

## Troubleshooting
- NaN loss: Reduce LR 10x, check data
- OOM: Reduce batch_size, increase grad_accum
- Slow: Enable CUDA, use KV-cache
- Overfitting: Reduce data, increase dropout, early stop
