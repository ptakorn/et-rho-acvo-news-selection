# Reproducibility notes and unresolved manuscript details

This implementation follows the supplied manuscript's high-level algorithms. It is not asserted to reproduce the exact reported table values because the paper does not provide every implementation detail and contains internal inconsistencies.

## Settings stated in the manuscript

- Static split: 80% training/validation pool and 20% held-out test; five-fold CV within the pool.
- Reported cluster counts: Indian 8, Reuters 13, Article 10, Financial 8, Variation 12, Microsoft 8, Thai 14.
- TPE search: Word2Vec vector size 50–200 in steps of 10, window 3–10, and CBOW/Skip-gram; maximum 1,000 trials and early stopping after 10 trials without improvement.
- ACVO: initial 80% and remaining 20% as a five-batch stream; both trigger thresholds are 10%; vocabulary is accumulated across batches; batch K-means is independent; the confidence baseline is initialized in-sample and refreshed in-sample after retraining.
- Results: 30 runs; mean and standard deviation; percentile-bootstrap 95% intervals; paired Wilcoxon signed-rank comparisons across seven dataset-level scores with Holm adjustment.
- Runtime reported from the thesis: Python 3.7.6 and scikit-learn 0.24.2. This code targets current Python and library APIs and can produce different numeric values.

## Defaults introduced by this code

The manuscript does not give all values required for execution. This implementation sets TF-IDF maximum features to 20,000, SVD components to 100, LDA topics to 20, LDA iterations to 20, Word2Vec epochs to 10 and min_count to 1, RF trees to 300, K-means n_init to 10, and macro-averaged precision/recall/F1. These are operational defaults, not recovered paper settings. Adjust them only after checking the original experiment records.

## Items to reconcile before claiming exact reproduction

1. The manuscript gives seeds 1–30 in one paragraph and 1000–1029 in another. The config uses 1000–1029, following the detailed evaluation paragraph.
2. The averaging convention for Precision, Recall, and F1 in Tables 2–8 is not stated. This implementation uses macro averaging.
3. Exact SVD dimension, LDA topic count/settings, Word2Vec epochs/min_count, RF settings, and K-means initialization count are not fully stated.
4. Verify the exact dataset URLs, versions, row filters, duplicate removal, title fields, and source licenses. The manuscript's generic “Kaggle” descriptions are not sufficient for exact retrieval.
5. Verify whether the static baselines use the same feature-only representation, cluster reference, RF settings, and splits as ET-RHO. This package currently implements ET-RHO and ACVO, not every baseline from the manuscript.
6. The manuscript does not make a separate ground-truth-labeled test set available. Static scores here are pseudo-label agreement. They must not be described as independent semantic accuracy.
7. ACVO runs K-means independently for each batch and does not align cluster IDs or filter pseudo-labels. ARI is used for permutation-invariant partition agreement. The method does not guarantee prevention of incorrect pseudo-label propagation.
8. For streams without trustworthy timestamps, this code shuffles rows deterministically per seed. The paper says stream order was fixed before evaluation but does not specify the exact order; substitute the archived order if available.
9. ACVO's confidence baseline is an in-sample mean maximum probability both initially and immediately after retraining, as stated in the manuscript. This may be optimistic.
10. The feature-permutation diagnostic checks distance preservation and matched-seed K-means partitions. A permutation alone should not be described as improving clustering quality.

## Public release checklist

- Add exact source URLs and dataset/version metadata without uploading data that cannot be redistributed.
- Preserve the original code and environment lockfile if available; compare outputs against this reference implementation.
- Record repository commit, OS, Python/library versions, hardware, random seeds, and exact run commands.
- Include run-level CSV files and the scripts/configuration that regenerate every reported table and figure.
- Add an independent labeled evaluation only if a valid ground-truth test set exists and its labels were kept out of model selection and retraining.
- Choose and add a software license only after confirming ownership and intended reuse terms.
