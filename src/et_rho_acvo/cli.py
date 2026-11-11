from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from et_rho_acvo.config import load_config
from et_rho_acvo.static import run_static
from et_rho_acvo.streaming import run_stream


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="UTF-8 CSV file with one title per row")
    parser.add_argument("--text-column", required=True, help="Column containing news titles")
    parser.add_argument("--clusters", required=True, type=int, help="K-means cluster count for this dataset")
    parser.add_argument("--config", default="configs/reproduction.yaml", help="YAML experiment configuration")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--trials", type=int, default=None, help="Override TPE trial limit (use a small value for a smoke run)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="et-rho-acvo", description="ET-RHO static and ACVO streaming experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser("static", help="Run repeated static ET-RHO evaluation")
    _add_shared(static_parser)
    static_parser.add_argument("--runs", type=int, default=None)
    static_parser.add_argument("--seed-start", type=int, default=None)

    stream_parser = subparsers.add_parser("stream", help="Run ACVO on sequential batches")
    _add_shared(stream_parser)
    stream_parser.add_argument("--runs", type=int, default=None)
    stream_parser.add_argument("--seed-start", type=int, default=None)
    stream_parser.add_argument("--batches", type=int, default=None)
    stream_parser.add_argument("--timestamp-column", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    exp_cfg = config.get("experiment", {})
    run_count = int(args.runs if args.runs is not None else exp_cfg.get("repetitions", 30))
    seed_start = int(args.seed_start if args.seed_start is not None else exp_cfg.get("random_seed_start", 1000))

    if args.command == "static":
        result = run_static(
            args.input, args.text_column, args.clusters, config, output,
            seed_start=seed_start, runs=run_count, trials=args.trials,
        )
        print(result.to_string(index=False))
    else:
        batch_results = []
        for seed in range(seed_start, seed_start + run_count):
            batch_results.append(run_stream(
                args.input, args.text_column, args.clusters, config, output,
                seed=seed, timestamp_column=args.timestamp_column,
                batches=args.batches, trials=args.trials,
            ))
        combined = pd.concat(batch_results, ignore_index=True)
        combined.to_csv(output / "stream_all_runs.csv", index=False)
        print(combined.groupby("batch")[["ari", "mean_max_probability", "confidence_drop", "vocabulary_novelty"]].mean())


if __name__ == "__main__":
    main()
