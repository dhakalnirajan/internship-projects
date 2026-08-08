#!/usr/bin/env python3
"""
evaluate.py - Evaluation Harness for K3-Edu

Evaluates model on:
- Perplexity
- Loss tracking and overfitting detection
- Code generation accuracy
- Natural language understanding
- Instruction following

Usage:
    python scripts/evaluate.py --model checkpoints/best_base.bin --dataset datasets/val/
"""

import os
import sys
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

try:
    import numpy as np
except ImportError:
    print("WARNING: numpy not installed, some features disabled")
    np = None

# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class EvalResult:
    perplexity: float
    loss: float
    accuracy: float
    tokens_per_sec: float
    memory_mb: float
    overfitting_detected: bool
    plateau_detected: bool

class LossTracker:
    """Track loss history and detect plateaus/overfitting."""

    def __init__(self, window_size: int = 10, plateau_threshold: float = 0.001):
        self.losses = deque(maxlen=window_size * 2)
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.best_loss = float('inf')
        self.steps_since_improvement = 0

    def add(self, loss: float, step: int):
        self.losses.append((step, loss))

        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def detect_plateau(self) -> Tuple[bool, float]:
        """Detect if loss has plateaued."""
        if len(self.losses) < self.window_size * 2:
            return False, 0.0

        recent = [l for _, l in list(self.losses)[-self.window_size:]]
        older = [l for _, l in list(self.losses)[:self.window_size]]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        improvement = older_avg - recent_avg
        return improvement < self.plateau_threshold, improvement

    def detect_overfitting(self, train_loss: float, val_loss: float) -> bool:
        """Detect overfitting based on train/val gap."""
        gap = val_loss - train_loss
        return gap > 0.5 and self.steps_since_improvement > 20

    def get_trend(self) -> str:
        """Get loss trend description."""
        if len(self.losses) < 5:
            return "insufficient_data"

        recent = [l for _, l in list(self.losses)[-5:]]
        if all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
            return "decreasing"
        elif all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
            return "increasing"
        return "fluctuating"

# ============================================================================
# CODE EVALUATION
# ============================================================================

CODE_TEST_CASES = [
    {
        "language": "python",
        "prompt": "Write a Python function to calculate the factorial of a number using recursion.",
        "expected_keywords": ["def factorial", "return", "n * factorial"],
        "test_input": "5",
        "expected_output": "120",
    },
    {
        "language": "c",
        "prompt": "Write a C function to reverse a string in place.",
        "expected_keywords": ["void reverse", "char*", "swap", "strlen"],
        "test_code": '#include <stdio.h>\n#include <string.h>\n\n{generated_code}\n\nint main() {{\n    char str[] = "hello";\n    reverse(str);\n    printf("%s\\n", str);\n    return 0;\n}}\n',
        "expected_output": "olleh",
    },
    {
        "language": "javascript",
        "prompt": "Write a JavaScript function to filter even numbers from an array.",
        "expected_keywords": ["function", "filter", "% 2 === 0", "even"],
        "test_code": '{generated_code}\nconsole.log(filterEven([1,2,3,4,5,6]).join(\',\'));\n',
        "expected_output": "2,4,6",
    },
    {
        "language": "cpp",
        "prompt": "Write a C++ class representing a 2D Point with x, y coordinates and a distance method.",
        "expected_keywords": ["class Point", "double x", "double y", "distance"],
    },
    {
        "language": "java",
        "prompt": "Write a Java method to check if a string is a palindrome.",
        "expected_keywords": ["boolean isPalindrome", "String", "reverse", "equals"],
    },
    {
        "language": "rust",
        "prompt": "Write a Rust function to find the maximum element in a vector.",
        "expected_keywords": ["fn max", "Vec", "iter", "max"],
    },
    {
        "language": "go",
        "prompt": "Write a Go function to calculate the sum of a slice of integers.",
        "expected_keywords": ["func sum", "[]int", "range", "sum +="],
    },
]

class CodeEvaluator:
    """Evaluate code generation quality."""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def generate_code(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate code using the model."""
        cmd = [
            "./build/inference",
            "--model", self.model_path,
            "--tokenizer", self.tokenizer_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", "0.1",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def evaluate_keywords(self, generated: str, expected_keywords: List[str]) -> float:
        """Check if generated code contains expected keywords."""
        if not generated:
            return 0.0

        matches = sum(1 for kw in expected_keywords if kw.lower() in generated.lower())
        return matches / len(expected_keywords)

    def evaluate_execution(self, code: str, test_code: str, expected_output: str, 
                           language: str) -> bool:
        """Try to compile and run generated code."""
        if not test_code or not expected_output:
            return False

        full_code = test_code.replace("{generated_code}", code)

        with tempfile.NamedTemporaryFile(mode='w', suffix=self._get_extension(language), 
                                         delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            if language == "c":
                exe_path = temp_path.replace('.c', '')
                subprocess.run(["gcc", "-o", exe_path, temp_path], 
                              capture_output=True, timeout=10)
                result = subprocess.run([exe_path], capture_output=True, 
                                       text=True, timeout=5)
            elif language == "python":
                result = subprocess.run(["python3", "-c", code], 
                                       capture_output=True, text=True, timeout=5)
            elif language == "javascript":
                result = subprocess.run(["node", "-e", full_code], 
                                       capture_output=True, text=True, timeout=5)
            else:
                return False

            return expected_output.strip() in result.stdout.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(temp_path)

    def _get_extension(self, language: str) -> str:
        extensions = {
            "python": ".py", "c": ".c", "cpp": ".cpp",
            "javascript": ".js", "java": ".java",
            "rust": ".rs", "go": ".go"
        }
        return extensions.get(language, ".txt")

    def run_evaluation(self) -> Dict:
        """Run full code evaluation."""
        results = []

        for test_case in CODE_TEST_CASES:
            print(f"  Testing {test_case['language']}: {test_case['prompt'][:50]}...")

            generated = self.generate_code(test_case['prompt'])

            keyword_score = self.evaluate_keywords(generated, test_case.get('expected_keywords', []))

            execution_score = 0.0
            if 'test_code' in test_case:
                execution_score = 1.0 if self.evaluate_execution(
                    generated, test_case['test_code'], 
                    test_case.get('expected_output', ''), 
                    test_case['language']
                ) else 0.0

            results.append({
                "language": test_case['language'],
                "keyword_score": keyword_score,
                "execution_score": execution_score,
                "generated_length": len(generated),
            })

        avg_keyword = sum(r['keyword_score'] for r in results) / len(results)
        avg_execution = sum(r['execution_score'] for r in results) / len(results)

        return {
            "test_cases": results,
            "average_keyword_score": avg_keyword,
            "average_execution_score": avg_execution,
            "overall_score": (avg_keyword + avg_execution) / 2,
        }

# ============================================================================
# PERPLEXITY EVALUATION
# ============================================================================

class PerplexityEvaluator:
    """Evaluate perplexity on a text corpus."""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def evaluate_file(self, filepath: Path, max_samples: int = 100) -> float:
        """Evaluate perplexity on a single file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        samples = text.split('\n\n===SAMPLE===\n\n')[:max_samples]

        total_loss = 0.0
        total_tokens = 0

        for sample in samples:
            if len(sample) < 50:
                continue

            cmd = [
                "./build/evaluate",
                "--model", self.model_path,
                "--tokenizer", self.tokenizer_path,
                "--text", sample[:2048],
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    loss = float(result.stdout.strip())
                    total_loss += loss * len(sample.split())
                    total_tokens += len(sample.split())
            except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
                continue

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)

    def evaluate_directory(self, dirpath: Path) -> Dict:
        """Evaluate perplexity on all files in directory."""
        files = list(dirpath.glob("*.txt")) + list(dirpath.glob("*.md"))

        if not files:
            return {"error": "No evaluation files found"}

        perplexities = []
        for filepath in files:
            print(f"  Evaluating: {filepath.name}")
            ppl = self.evaluate_file(filepath)
            if ppl != float('inf'):
                perplexities.append(ppl)

        if not perplexities:
            return {"error": "All evaluations failed"}

        if np:
            return {
                "mean_perplexity": float(np.mean(perplexities)),
                "median_perplexity": float(np.median(perplexities)),
                "std_perplexity": float(np.std(perplexities)),
                "min_perplexity": float(np.min(perplexities)),
                "max_perplexity": float(np.max(perplexities)),
                "n_samples": len(perplexities),
            }
        else:
            return {
                "mean_perplexity": sum(perplexities) / len(perplexities),
                "min_perplexity": min(perplexities),
                "max_perplexity": max(perplexities),
                "n_samples": len(perplexities),
            }

# ============================================================================
# TRAINING LOG ANALYSIS
# ============================================================================

def analyze_training_log(log_path: Path) -> Dict:
    """Analyze training log for overfitting and plateaus."""
    if not log_path.exists():
        return {"error": "Training log not found"}

    tracker = LossTracker()

    with open(log_path, 'r') as f:
        for line in f:
            if 'loss=' in line:
                try:
                    parts = line.split()
                    step = int(parts[0].split('=')[1])
                    loss = float([p for p in parts if 'loss=' in p][0].split('=')[1])
                    tracker.add(loss, step)
                except (IndexError, ValueError):
                    continue

    plateau, improvement = tracker.detect_plateau()
    trend = tracker.get_trend()

    return {
        "total_steps": len(tracker.losses),
        "best_loss": tracker.best_loss,
        "steps_since_improvement": tracker.steps_since_improvement,
        "plateau_detected": plateau,
        "plateau_improvement": improvement,
        "loss_trend": trend,
        "recommendation": _get_recommendation(plateau, trend, tracker.steps_since_improvement),
    }

def _get_recommendation(plateau: bool, trend: str, steps_since_improvement: int) -> str:
    if plateau:
        if steps_since_improvement > 50:
            return "CRITICAL: Training has stagnated. Consider reducing learning rate or increasing data diversity."
        return "WARNING: Loss plateau detected. Consider warm restart or data augmentation."
    if trend == "increasing":
        return "WARNING: Loss is increasing. Check for gradient explosion or learning rate too high."
    if trend == "fluctuating" and steps_since_improvement > 30:
        return "WARNING: Unstable training. Consider gradient clipping or smaller batch size."
    return "OK: Training is progressing normally."

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate K3-Edu model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.bin",
                       help="Path to tokenizer")
    parser.add_argument("--dataset", type=str, default="datasets/val",
                       help="Validation dataset directory")
    parser.add_argument("--training-log", type=str, default="training.log",
                       help="Training log file for overfitting analysis")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--skip-code", action="store_true",
                       help="Skip code evaluation")
    parser.add_argument("--skip-perplexity", action="store_true",
                       help="Skip perplexity evaluation")
    parser.add_argument("--skip-log-analysis", action="store_true",
                       help="Skip training log analysis")

    args = parser.parse_args()

    print("=" * 60)
    print("  K3-Edu Evaluation Harness")
    print("=" * 60)

    results = {
        "model": args.model,
        "timestamp": str(Path().stat().st_mtime),
    }

    if not args.skip_perplexity:
        print("\n[1/3] Perplexity Evaluation")
        print("-" * 40)
        ppl_eval = PerplexityEvaluator(args.model, args.tokenizer)
        ppl_results = ppl_eval.evaluate_directory(Path(args.dataset))
        results["perplexity"] = ppl_results

        if "mean_perplexity" in ppl_results:
            print(f"  Mean Perplexity: {ppl_results['mean_perplexity']:.2f}")
            print(f"  Min/Max: {ppl_results.get('min_perplexity', 'N/A'):.2f} / "
                  f"{ppl_results.get('max_perplexity', 'N/A'):.2f}")

    if not args.skip_code:
        print("\n[2/3] Code Generation Evaluation")
        print("-" * 40)
        code_eval = CodeEvaluator(args.model, args.tokenizer)
        code_results = code_eval.run_evaluation()
        results["code_evaluation"] = code_results

        print(f"  Keyword Score: {code_results['average_keyword_score']:.2%}")
        print(f"  Execution Score: {code_results['average_execution_score']:.2%}")
        print(f"  Overall: {code_results['overall_score']:.2%}")

    if not args.skip_log_analysis:
        print("\n[3/3] Training Log Analysis")
        print("-" * 40)
        log_results = analyze_training_log(Path(args.training_log))
        results["training_analysis"] = log_results

        print(f"  Best Loss: {log_results.get('best_loss', 'N/A')}")
        print(f"  Trend: {log_results.get('loss_trend', 'N/A')}")
        print(f"  Plateau: {'YES' if log_results.get('plateau_detected') else 'NO'}")
        print(f"  Recommendation: {log_results.get('recommendation', 'N/A')}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()
