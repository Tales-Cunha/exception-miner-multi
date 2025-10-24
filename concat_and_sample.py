#!/usr/bin/env python3
"""Utilities to merge parser outputs and build balanced samples."""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def list_csvs(source_dir: Path, recursive: bool) -> List[Path]:
    pattern = "*.csv"
    if recursive:
        files = sorted(path for path in source_dir.rglob(pattern) if path.is_file())
    else:
        files = sorted(path for path in source_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found under {source_dir}")
    return files


def concat_csvs(files: List[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in files:
        frames.append(pd.read_csv(csv_path))
    combined = pd.concat(frames, ignore_index=True)
    return combined


def ensure_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Required column '{column}' not present in concatenated DataFrame")


def write_csv(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def build_sample(df: pd.DataFrame, column: str, sample_size: int, seed: int) -> pd.DataFrame:
    positive = df[df[column] == 1]
    negative = df[df[column] == 0]
    if len(positive) < sample_size or len(negative) < sample_size:
        raise ValueError(
            "Not enough rows to build the requested balanced sample: "
            f"need {sample_size} with {column} == 1 and {sample_size} with {column} == 0"
        )
    pos_sample = positive.sample(n=sample_size, random_state=seed)
    neg_sample = negative.sample(n=sample_size, random_state=seed + 1)
    merged = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return merged.sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate parser CSVs and build balanced subsets.")
    parser.add_argument("source_dir", type=Path, help="Directory containing CSV files to merge.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for CSV files.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("tmp/combined_dataset.csv"),
        help="Where to store the concatenated CSV (default: tmp/combined_dataset.csv).",
    )
    parser.add_argument(
        "--exceptions-output",
        type=Path,
        default=Path("tmp/combined_exceptions.csv"),
        help="Where to store rows with n_try_except == 1 (default: tmp/combined_exceptions.csv).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional count of rows to sample for each class of n_try_except.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("tmp/balanced_sample.csv"),
        help="Where to store the balanced sample when --sample-size is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = list_csvs(args.source_dir, args.recursive)
    combined = concat_csvs(csv_files)
    write_csv(combined, args.combined_output)
    print(f"Loaded {len(csv_files)} CSV files.")
    print(f"Combined rows: {len(combined)} -> {args.combined_output}")

    ensure_column(combined, "n_try_except")
    exceptions = combined[combined["n_try_except"] == 1]
    write_csv(exceptions, args.exceptions_output)
    print(f"Rows with n_try_except == 1: {len(exceptions)} -> {args.exceptions_output}")

    if args.sample_size:
        balanced = build_sample(
            combined,
            column="n_try_except",
            sample_size=args.sample_size,
            seed=args.seed,
        )
        write_csv(balanced, args.sample_output)
        print(
            "Balanced sample rows: "
            f"{len(balanced)} (each class {args.sample_size}) -> {args.sample_output}"
        )


if __name__ == "__main__":
    main()