from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def make_uniform_seed_pool(seed_pool: dict[str, list[str]], target_n: int | None = None) -> dict[str, list[str]]:
    """Trim each emotion's seed list to the same length.

    If target_n is None, the shortest available list length is used.
    """
    if not seed_pool:
        return {}

    counts = {emotion: len(words) for emotion, words in seed_pool.items()}
    uniform_n = min(counts.values()) if target_n is None else int(target_n)

    lacking = [emotion for emotion, count in counts.items() if count < uniform_n]
    if lacking:
        raise ValueError(
            f"Some emotions do not have {uniform_n} seeds: "
            + ", ".join(f"{emotion}={counts[emotion]}" for emotion in lacking)
        )

    return {emotion: list(words[:uniform_n]) for emotion, words in seed_pool.items()}


def generation_report(
    df: pd.DataFrame,
    emotion_col: str = "primary_emotion",
    seed_col: str = "seed_word",
) -> pd.DataFrame:
    """Summarize how many seeds and sentences are available per emotion."""
    if df.empty:
        return pd.DataFrame(
            columns=[emotion_col, "n_rows", "n_unique_seeds", "min_sentences_per_seed", "max_sentences_per_seed"]
        )

    per_seed = (
        df.groupby([emotion_col, seed_col], sort=True)
        .size()
        .rename("n_sentences")
        .reset_index()
    )
    report = (
        per_seed.groupby(emotion_col, sort=True)["n_sentences"]
        .agg(n_unique_seeds="size", min_sentences_per_seed="min", max_sentences_per_seed="max")
        .reset_index()
    )
    row_counts = df.groupby(emotion_col, sort=True).size().rename("n_rows").reset_index()
    return report.merge(row_counts, on=emotion_col)[
        [emotion_col, "n_rows", "n_unique_seeds", "min_sentences_per_seed", "max_sentences_per_seed"]
    ]


def rebalance_generated_dataset(
    df: pd.DataFrame,
    target_seeds_per_emotion: int | None = None,
    target_sentences_per_seed: int | None = None,
    emotion_col: str = "primary_emotion",
    seed_col: str = "seed_word",
    sentence_col: str = "generated_sentence",
) -> pd.DataFrame:
    """Keep an exactly balanced subset from generated rows.

    Rules:
    - Every retained emotion has the same number of seeds.
    - Every retained seed has the same number of sentences.
    - Seed order follows first appearance in the input dataframe.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    work[emotion_col] = work[emotion_col].astype(str)
    work[seed_col] = work[seed_col].astype(str).str.lower()
    work[sentence_col] = work[sentence_col].astype(str).str.strip()
    work = work[work[sentence_col] != ""].copy()
    work["_row_order"] = np.arange(len(work))

    per_seed = (
        work.groupby([emotion_col, seed_col], sort=False)
        .size()
        .rename("n_sentences")
        .reset_index()
    )

    target_sentences = (
        int(per_seed["n_sentences"].min())
        if target_sentences_per_seed is None
        else int(target_sentences_per_seed)
    )

    eligible = per_seed[per_seed["n_sentences"] >= target_sentences].copy()
    eligible_seed_counts = eligible.groupby(emotion_col, sort=True).size()
    target_seeds = (
        int(eligible_seed_counts.min())
        if target_seeds_per_emotion is None
        else int(target_seeds_per_emotion)
    )

    short = eligible_seed_counts[eligible_seed_counts < target_seeds]
    if not short.empty:
        raise ValueError(
            f"Some emotions do not have {target_seeds} complete seeds: "
            + ", ".join(f"{emotion}={count}" for emotion, count in short.items())
        )

    first_seen = (
        work.groupby([emotion_col, seed_col], sort=False)["_row_order"]
        .min()
        .rename("first_seen")
        .reset_index()
    )
    eligible = eligible.merge(first_seen, on=[emotion_col, seed_col], how="left")
    eligible = eligible.sort_values([emotion_col, "first_seen", seed_col], kind="stable")

    kept_pairs = (
        eligible.groupby(emotion_col, sort=False)
        .head(target_seeds)[[emotion_col, seed_col]]
        .copy()
    )
    kept_pairs["_keep"] = 1

    balanced = work.merge(kept_pairs, on=[emotion_col, seed_col], how="inner")
    balanced = balanced.sort_values([emotion_col, seed_col, "_row_order"], kind="stable")
    balanced = (
        balanced.groupby([emotion_col, seed_col], sort=False)
        .head(target_sentences)
        .copy()
    )

    balanced = balanced.drop(columns=["_row_order", "_keep"]).reset_index(drop=True)
    if "id" in balanced.columns:
        balanced["id"] = np.arange(len(balanced))
    return balanced


def merge_raw_checkpoint_files(
    checkpoint_dir: str | Path,
    emotions: Iterable[str] | None = None,
) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir)
    if emotions is None:
        paths = sorted(checkpoint_dir.glob("raw_*.csv"))
    else:
        paths = [checkpoint_dir / f"raw_{emotion}.csv" for emotion in emotions]

    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists() and path.stat().st_size > 1:
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if "id" in merged.columns:
        merged["id"] = np.arange(len(merged))
    return merged
