#!/usr/bin/env python3
"""
prepare_data.py - Dataset Preprocessing for K3-Edu

Downloads and prepares training datasets from HuggingFace.
Supports natural language (English, Nepali) and programming languages
(Python, C, C++, Java, JavaScript, Rust, Go).

Usage:
    python scripts/prepare_data.py --output datasets/ --size 5GB
"""

import os
import sys
import argparse
import random
import json
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: Please install required packages:")
    print("  pip install datasets transformers")
    sys.exit(1)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

NATURAL_LANGUAGE_DATASETS = {
    "english": [
        ("wikitext", "wikitext-103-raw-v1"),
        ("openwebtext", None),
        ("c4", "en"),
    ],
    "nepali": [
        ("CohereForAI/aya_collection", None),  # Multilingual
        ("mc4", "ne"),  # Nepali subset of mC4
    ]
}

CODE_DATASETS = {
    "python": [
        ("codeparrot/github-code", "Python"),
        ("bigcode/the-stack", "Python"),
    ],
    "c_cpp": [
        ("codeparrot/github-code", "C"),
        ("codeparrot/github-code", "C++"),
    ],
    "javascript": [
        ("codeparrot/github-code", "JavaScript"),
    ],
    "java": [
        ("codeparrot/github-code", "Java"),
    ],
    "rust": [
        ("codeparrot/github-code", "Rust"),
    ],
    "go": [
        ("codeparrot/github-code", "Go"),
    ],
}

INSTRUCTION_DATASETS = [
    ("tatsu-lab/alpaca", None),
    ("Open-Orca/OpenOrca", None),
    ("HuggingFaceH4/CodeAlpaca-20k", None),
    ("iamtarun/python_code_instructions_18k_alpaca", None),
]

# ============================================================================
# UTILITIES
# ============================================================================

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def format_size(bytes_size: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"

def shuffle_and_split(texts: List[str], val_ratio: float = 0.05) -> tuple:
    """Shuffle texts and split into train/val."""
    random.seed(42)
    random.shuffle(texts)
    split_idx = int(len(texts) * (1 - val_ratio))
    return texts[:split_idx], texts[split_idx:]

def save_texts(texts: List[str], output_dir: Path, prefix: str, max_file_size_mb: int = 50):
    """Save texts to .txt files, splitting if too large."""
    ensure_dir(output_dir)

    current_file = 0
    current_size = 0
    current_texts = []
    max_bytes = max_file_size_mb * 1024 * 1024

    for text in texts:
        text_bytes = len(text.encode('utf-8'))

        if current_size + text_bytes > max_bytes and current_texts:
            # Save current batch
            filepath = output_dir / f"{prefix}_{current_file:04d}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n\n===SAMPLE===\n\n'.join(current_texts))
            print(f"  Saved: {filepath} ({format_size(current_size)})")

            current_file += 1
            current_size = 0
            current_texts = []

        current_texts.append(text)
        current_size += text_bytes

    # Save remaining
    if current_texts:
        filepath = output_dir / f"{prefix}_{current_file:04d}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n\n===SAMPLE===\n\n'.join(current_texts))
        print(f"  Saved: {filepath} ({format_size(current_size)})")

def download_natural_language(output_dir: Path, target_size_gb: float, languages: List[str]):
    """Download natural language datasets."""
    print(f"\n[1/3] Downloading natural language datasets...")
    print(f"  Target size: {target_size_gb:.1f} GB")
    print(f"  Languages: {', '.join(languages)}")

    all_texts = []

    for lang in languages:
        if lang not in NATURAL_LANGUAGE_DATASETS:
            print(f"  WARNING: Unknown language '{lang}', skipping")
            continue

        for dataset_name, config_name in NATURAL_LANGUAGE_DATASETS[lang]:
            try:
                print(f"  Loading: {dataset_name}" + (f" ({config_name})" if config_name else ""))

                if config_name:
                    ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
                else:
                    ds = load_dataset(dataset_name, split="train", streaming=True)

                # Collect texts
                texts = []
                total_bytes = 0
                target_bytes = int(target_size_gb * 1024 * 1024 * 1024 / len(languages))

                for item in ds:
                    text = item.get('text', item.get('content', ''))
                    if text and len(text) > 100:
                        texts.append(text)
                        total_bytes += len(text.encode('utf-8'))

                        if total_bytes >= target_bytes:
                            break

                print(f"    Collected {len(texts)} samples ({format_size(total_bytes)})")
                all_texts.extend(texts)

            except Exception as e:
                print(f"    ERROR loading {dataset_name}: {e}")
                continue

    # Shuffle and save
    train_texts, val_texts = shuffle_and_split(all_texts)

    print(f"\n  Saving {len(train_texts)} train samples...")
    save_texts(train_texts, output_dir / "train", "nl")

    print(f"  Saving {len(val_texts)} validation samples...")
    save_texts(val_texts, output_dir / "val", "nl")

def download_code(output_dir: Path, target_size_gb: float, languages: List[str]):
    """Download code datasets."""
    print(f"\n[2/3] Downloading code datasets...")
    print(f"  Target size: {target_size_gb:.1f} GB")
    print(f"  Languages: {', '.join(languages)}")

    all_texts = []

    for lang in languages:
        if lang not in CODE_DATASETS:
            print(f"  WARNING: Unknown language '{lang}', skipping")
            continue

        for dataset_name, config_name in CODE_DATASETS[lang]:
            try:
                print(f"  Loading: {dataset_name}" + (f" ({config_name})" if config_name else ""))

                if config_name:
                    ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
                else:
                    ds = load_dataset(dataset_name, split="train", streaming=True)

                texts = []
                total_bytes = 0
                target_bytes = int(target_size_gb * 1024 * 1024 * 1024 / len(languages))

                for item in ds:
                    text = item.get('content', item.get('code', ''))
                    if text and len(text) > 50:
                        # Add language marker
                        marked_text = f"# Language: {lang.upper()}\n{text}"
                        texts.append(marked_text)
                        total_bytes += len(marked_text.encode('utf-8'))

                        if total_bytes >= target_bytes:
                            break

                print(f"    Collected {len(texts)} samples ({format_size(total_bytes)})")
                all_texts.extend(texts)

            except Exception as e:
                print(f"    ERROR loading {dataset_name}: {e}")
                continue

    train_texts, val_texts = shuffle_and_split(all_texts)

    print(f"\n  Saving {len(train_texts)} train samples...")
    save_texts(train_texts, output_dir / "train", "code")

    print(f"  Saving {len(val_texts)} validation samples...")
    save_texts(val_texts, output_dir / "val", "code")

def download_instructions(output_dir: Path, target_size_gb: float):
    """Download instruction-tuning datasets."""
    print(f"\n[3/3] Downloading instruction datasets...")
    print(f"  Target size: {target_size_gb:.1f} GB")

    all_instructions = []

    for dataset_name, config_name in INSTRUCTION_DATASETS:
        try:
            print(f"  Loading: {dataset_name}")

            if config_name:
                ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)

            instructions = []
            total_bytes = 0
            target_bytes = int(target_size_gb * 1024 * 1024 * 1024 / len(INSTRUCTION_DATASETS))

            for item in ds:
                # Format as instruction
                instruction = item.get('instruction', item.get('question', ''))
                input_text = item.get('input', '')
                output_text = item.get('output', item.get('response', ''))

                if instruction and output_text:
                    formatted = f"<|im_start|>user\n{instruction}"
                    if input_text:
                        formatted += f"\n{input_text}"
                    formatted += f"<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>\n"

                    instructions.append(formatted)
                    total_bytes += len(formatted.encode('utf-8'))

                    if total_bytes >= target_bytes:
                        break

            print(f"    Collected {len(instructions)} instructions ({format_size(total_bytes)})")
            all_instructions.extend(instructions)

        except Exception as e:
            print(f"    ERROR loading {dataset_name}: {e}")
            continue

    train_instr, val_instr = shuffle_and_split(all_instructions)

    print(f"\n  Saving {len(train_instr)} train instructions...")
    save_texts(train_instr, output_dir / "train", "instr")

    print(f"  Saving {len(val_instr)} validation instructions...")
    save_texts(val_instr, output_dir / "val", "instr")

def create_dataset_info(output_dir: Path, args):
    """Create dataset metadata file."""
    info = {
        "name": "K3-Edu Training Dataset",
        "version": "1.0",
        "natural_languages": args.languages,
        "programming_languages": args.code_languages,
        "total_size_gb": args.total_size,
        "nl_ratio": args.nl_ratio,
        "code_ratio": args.code_ratio,
        "instruction_ratio": args.instruction_ratio,
        "created": str(Path().stat().st_mtime),
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset info saved to: {info_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets for K3-Edu")
    parser.add_argument("--output", type=str, default="datasets",
                       help="Output directory for datasets")
    parser.add_argument("--total-size", type=float, default=5.0,
                       help="Total dataset size in GB (default: 5.0)")
    parser.add_argument("--nl-ratio", type=float, default=0.4,
                       help="Ratio of natural language data (default: 0.4)")
    parser.add_argument("--code-ratio", type=float, default=0.4,
                       help="Ratio of code data (default: 0.4)")
    parser.add_argument("--instruction-ratio", type=float, default=0.2,
                       help="Ratio of instruction data (default: 0.2)")
    parser.add_argument("--languages", nargs="+", default=["english", "nepali"],
                       help="Natural languages to include")
    parser.add_argument("--code-languages", nargs="+", 
                       default=["python", "c_cpp", "javascript", "java", "rust", "go"],
                       help="Programming languages to include")
    parser.add_argument("--skip-nl", action="store_true",
                       help="Skip natural language datasets")
    parser.add_argument("--skip-code", action="store_true",
                       help="Skip code datasets")
    parser.add_argument("--skip-instructions", action="store_true",
                       help="Skip instruction datasets")

    args = parser.parse_args()

    output_dir = ensure_dir(args.output)

    print("=" * 60)
    print("  K3-Edu Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total target size: {args.total_size} GB")

    # Calculate sizes
    nl_size = args.total_size * args.nl_ratio if not args.skip_nl else 0
    code_size = args.total_size * args.code_ratio if not args.skip_code else 0
    instr_size = args.total_size * args.instruction_ratio if not args.skip_instructions else 0

    print(f"  Natural language: {nl_size:.1f} GB")
    print(f"  Code: {code_size:.1f} GB")
    print(f"  Instructions: {instr_size:.1f} GB")

    # Download datasets
    if nl_size > 0:
        download_natural_language(output_dir, nl_size, args.languages)

    if code_size > 0:
        download_code(output_dir, code_size, args.code_languages)

    if instr_size > 0:
        download_instructions(output_dir, instr_size)

    # Save metadata
    create_dataset_info(output_dir, args)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Files saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Train tokenizer: ./build/train_tokenizer")
    print("  2. Train base model: ./build/train_base")
    print("  3. Fine-tune: ./build/train_inst")
    print("=" * 60)

if __name__ == "__main__":
    main()
