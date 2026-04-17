from __future__ import annotations

import argparse
from pathlib import Path

from balance_utils import generation_report, merge_raw_checkpoint_files, rebalance_generated_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a balanced synthetic dataset from raw checkpoint CSV files.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("lab7/data/synthetic_checkpoints_gemma4_e2b_v2"),
        help="Directory containing raw_<emotion>.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lab7/data/synthetic_emotion_dataset_gemma4_e2b_balanced.csv"),
        help="Output CSV path for the balanced dataset.",
    )
    parser.add_argument(
        "--target-seeds-per-emotion",
        type=int,
        default=None,
        help="Exact number of seeds to keep per emotion. Default: minimum complete count across emotions.",
    )
    parser.add_argument(
        "--target-sentences-per-seed",
        type=int,
        default=None,
        help="Exact number of sentences to keep per seed. Default: minimum count found in the data.",
    )
    args = parser.parse_args()

    raw_df = merge_raw_checkpoint_files(args.checkpoint_dir)
    if raw_df.empty:
        raise SystemExit(f"No raw checkpoint CSV files found in {args.checkpoint_dir}")

    print("Raw dataset:")
    print(generation_report(raw_df).to_string(index=False))
    print()

    balanced_df = rebalance_generated_dataset(
        raw_df,
        target_seeds_per_emotion=args.target_seeds_per_emotion,
        target_sentences_per_seed=args.target_sentences_per_seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(args.output, index=False)

    print("Balanced dataset:")
    print(generation_report(balanced_df).to_string(index=False))
    print()
    print(f"Saved balanced CSV to: {args.output}")


if __name__ == "__main__":
    main()
