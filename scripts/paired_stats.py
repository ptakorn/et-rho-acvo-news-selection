"""Compare dataset-level method scores using paired Wilcoxon and Holm correction."""

import argparse

import pandas as pd

from et_rho_acvo.statistics import compare_dataset_methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with one row per dataset and method")
    parser.add_argument("--reference", default="ET-RHO")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scores = pd.read_csv(args.input)
    result = compare_dataset_methods(scores, args.reference, args.metric)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
