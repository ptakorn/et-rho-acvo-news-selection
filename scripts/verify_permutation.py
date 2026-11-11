"""Verify distance and matched-seed K-means invariance to a global feature permutation."""

import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npy", required=True, help="Dense feature matrix saved with numpy.save")
    parser.add_argument("--clusters", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    x = np.load(args.input_npy)
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix.")
    permutation = np.random.default_rng(args.seed).permutation(x.shape[1])
    x_permuted = x[:, permutation]
    original_distances = pairwise_distances(x, metric="euclidean")
    permuted_distances = pairwise_distances(x_permuted, metric="euclidean")
    maximum_distance_error = float(np.max(np.abs(original_distances - permuted_distances)))

    original = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed).fit_predict(x)
    permuted = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed).fit_predict(x_permuted)
    matched_ari = float(adjusted_rand_score(original, permuted))
    print(f"max_abs_pairwise_distance_difference={maximum_distance_error:.12g}")
    print(f"matched_seed_partition_ARI={matched_ari:.12g}")
    print("A global column permutation preserves Euclidean distances; investigate any discrepancy beyond numerical tolerance.")


if __name__ == "__main__":
    main()
