#!/usr/bin/env python3
"""
Training Data Preparation
Takes augmented circuits and creates train/val/test splits as JSONL files.

Splits: 70% train / 15% val / 15% test (configurable)
Output format per line: {"description": "...", "circuit_qasm": "...", "category": "...", "source": "..."}
"""
import json
import random
import argparse
from pathlib import Path


def prepare_training_data(
    input_file="output/augmented_circuits.json",
    output_dir="output/training_data",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """Prepare data for QuantumGPT training."""

    print("=" * 70)
    print("TRAINING DATA PREPARATION")
    print("=" * 70)

    print(f"\n1. Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        circuits = json.load(f)

    print(f"   Total samples: {len(circuits):,}")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

    random.seed(seed)
    random.shuffle(circuits)
    print(f"   Shuffled with seed={seed}")

    n_total = len(circuits)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_circuits = circuits[:n_train]
    val_circuits = circuits[n_train:n_train + n_val]
    test_circuits = circuits[n_train + n_val:]

    print(f"\n2. Splits:")
    print(f"   Train: {len(train_circuits):,} samples ({100*len(train_circuits)/n_total:.1f}%)")
    print(f"   Val:   {len(val_circuits):,} samples ({100*len(val_circuits)/n_total:.1f}%)")
    print(f"   Test:  {len(test_circuits):,} samples ({100*len(test_circuits)/n_total:.1f}%)")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n3. Output directory: {output_path.absolute()}")

    def save_jsonl(circuits_list, filename):
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            for circuit in circuits_list:
                entry = {
                    "description":  circuit['description'],
                    "circuit_qasm": circuit['circuit_qasm'],
                    "category":     circuit.get('category', 'unknown'),
                    "source":       circuit.get('source', 'unknown'),
                }
                f.write(json.dumps(entry) + '\n')
        print(f"   {filename}: {len(circuits_list):,} entries")

    print(f"\n4. Saving JSONL files...")
    save_jsonl(train_circuits, "augmented_train.jsonl")
    save_jsonl(val_circuits,   "augmented_val.jsonl")
    save_jsonl(test_circuits,  "augmented_test.jsonl")

    print(f"\n5. Dataset Statistics:")

    def analyze_split(circuits_list, name):
        categories = {}
        sources = {}
        for c in circuits_list:
            cat = c.get('category', 'unknown')
            src = c.get('source', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            sources[src]    = sources.get(src, 0) + 1

        print(f"\n   {name}:")
        print(f"      Samples: {len(circuits_list):,}")
        print(f"      Unique categories: {len(categories)}")
        print(f"      Top 5 categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"        - {cat}: {count}")
        print(f"      Sources: {', '.join([f'{k}={v}' for k, v in sources.items()])}")

    analyze_split(train_circuits, "TRAIN")
    analyze_split(val_circuits,   "VAL")
    analyze_split(test_circuits,  "TEST")

    print(f"\n6. Token Estimation:")
    sample_size = min(200, len(circuits))
    sample_circuits = random.sample(circuits, sample_size)

    total_chars = 0
    for c in sample_circuits:
        formatted = (
            f"<|user|>{c['description']}<|end|>\n"
            f"<|assistant|>{c['circuit_qasm']}<|end|>"
        )
        total_chars += len(formatted)

    avg_chars  = total_chars / sample_size
    avg_tokens = avg_chars / 4
    total_tokens = avg_tokens * len(circuits)

    print(f"   Avg chars per formatted sample: {avg_chars:.0f}")
    print(f"   Avg tokens per sample:          {avg_tokens:.0f}")
    print(f"   Estimated total tokens:         {total_tokens:,.0f}")
    print(f"   Train tokens: ~{avg_tokens * len(train_circuits):,.0f}")
    print(f"   Val tokens:   ~{avg_tokens * len(val_circuits):,.0f}")
    print(f"   Test tokens:  ~{avg_tokens * len(test_circuits):,.0f}")

    print(f"\n7. Training Recommendations:")
    train_tokens = avg_tokens * len(train_circuits)
    tokens_per_step = 8 * 8 * 256  # micro_batch * grad_accum * block_size
    for iters in [1500, 2000, 2500]:
        epochs = (tokens_per_step * iters) / train_tokens
        print(f"   max_iters={iters}: ~{epochs:.1f} epochs")

    if total_tokens < 200_000:
        print(f"\n   CAUTION: Moderate dataset -- dropout 0.2, watch train/val gap")
    else:
        print(f"\n   Good dataset size -- dropout 0.15-0.2, standard training")

    metadata = {
        "total_samples":               len(circuits),
        "train_samples":               len(train_circuits),
        "val_samples":                 len(val_circuits),
        "test_samples":                len(test_circuits),
        "estimated_total_tokens":      int(total_tokens),
        "estimated_tokens_per_sample": int(avg_tokens),
        "train_ratio":                 train_ratio,
        "val_ratio":                   val_ratio,
        "test_ratio":                  test_ratio,
        "seed":                        seed,
        "input_file":                  input_file,
        "output_directory":            str(output_path),
    }

    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nReady to train!")
    print(f"Files in: {output_path}/")
    print(f"  augmented_train.jsonl")
    print(f"  augmented_val.jsonl")
    print(f"  augmented_test.jsonl")
    print(f"\nNext step: copy output/training_data/ into your quantumgpt-training repo")
    print(f"and run: python train.py")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Prepare quantum circuit training data")
    parser.add_argument("--input",  default="output/augmented_circuits.json",
                        help="Input augmented circuits JSON (default: output/augmented_circuits.json)")
    parser.add_argument("--output", default="output/training_data",
                        help="Output directory for JSONL files (default: output/training_data)")
    parser.add_argument("--train",  type=float, default=0.70)
    parser.add_argument("--val",    type=float, default=0.15)
    parser.add_argument("--test",   type=float, default=0.15)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    prepare_training_data(
        input_file=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
