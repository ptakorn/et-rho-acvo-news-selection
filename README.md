# ET-RHO and ACVO for News-Title Processing

Reference implementation for the manuscript **“Word Embedding Optimization with Clustering-Based Pseudo-Labels for News Selection in Online News Providers.”**

ET-RHO combines TF-IDF/SVD, Word2Vec, and LDA representations, tunes Word2Vec settings with TPE, derives training pseudo-labels from K-means, and trains a Random Forest (RF). ACVO processes sequential batches and triggers representation refitting and RF retraining when vocabulary novelty or a decline in mean maximum RF probability crosses a threshold.

## Important interpretation

The manuscript reports no human ground-truth labels for the seven datasets. Accordingly, the static Accuracy, Precision, Recall, and F1 scores in this implementation are **agreement with K-means pseudo-labels**, not semantic news-classification accuracy. The streaming ARI compares the pre-update RF partition with a separately fitted K-means partition for each batch. It is invariant to cluster-ID permutations, but it does not establish semantic accuracy or verified concept drift.

The manuscript does not specify several implementation details, and it contains two different seed ranges (1–30 and 1000–1029). This repository makes the selected defaults configurable and lists unresolved items in [`docs/reproducibility.md`](docs/reproducibility.md). Do not describe outputs from these defaults as exact reproductions of the published tables until the settings are reconciled with the original experiment records.

## Project layout

```text
configs/reproduction.yaml       Central experiment settings
data/README.md                  Dataset placement and access notes
data/demo/news_titles.csv       Small synthetic example (not paper data)
docs/reproducibility.md         Reported settings, explicit defaults, open issues
scripts/                        Commands for static, streaming, and permutation checks
src/et_rho_acvo/                 ET-RHO and ACVO implementation
```

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Thai text tokenization uses PyThaiNLP. English tokenization uses a Unicode word tokenizer. The demo CSV is synthetic and exists only to illustrate the input format.

## Quick run

From the repository root, run one small demonstration trial:

```bash
python scripts/run_static.py --input data/demo/news_titles.csv --text-column title --clusters 3 --runs 1 --trials 2 --output results/demo-static
```

Run the ACVO batch demonstration:

```bash
python scripts/run_stream.py --input data/demo/news_titles.csv --text-column title --clusters 2 --runs 1 --trials 2 --batches 2 --output results/demo-stream
```

The demo is too small to support scientific conclusions. For a paper dataset, place a CSV at a permitted local path with one title per row and pass its path to the scripts. Do not add restricted raw data to a public repository.

## Static ET-RHO experiment

```bash
python scripts/run_static.py --input /path/to/news.csv --text-column title --clusters 14 --runs 30 --seed-start 1000 --trials 1000 --output results/thai
```

The script makes an 80/20 random train/test split for each seed, tunes Word2Vec within five-fold CV on the 80% pool, refits the representation/K-means/RF on that pool, and evaluates RF predictions against pseudo-labels assigned by the fitted K-means model on the held-out 20%. It writes per-run scores and a summary CSV. Use the cluster count shown for the dataset in the manuscript's dataset table.

The summary includes run-level mean, sample standard deviation, and a 95% percentile-bootstrap interval. If you have a separate CSV with one F1 score per method and dataset (`dataset,method,f1`), the paired dataset-level Wilcoxon tests and Holm adjustment can be calculated with:

```bash
python scripts/paired_stats.py --input results/dataset_method_f1.csv --reference ET-RHO --metric f1 --output results/wilcoxon_holm.csv
```

## ACVO streaming experiment

```bash
python scripts/run_stream.py --input /path/to/news.csv --text-column title --clusters 14 --runs 30 --seed-start 1000 --trials 1000 --batches 5 --output results/thai-stream
```

If a trustworthy publication-time column exists, pass `--timestamp-column DATE_COLUMN` to sort rows chronologically. Otherwise, rows are deterministically shuffled per seed to create a simulated stream. The 80% initial portion trains the starting model; the remaining 20% is split into the requested number of batches. For each batch, ACVO records pre-update ARI, confidence, vocabulary novelty, trigger reason, and whether it refitted.

## Dataset cluster counts reported in the manuscript

| Dataset | Documents | Language | K |
|---|---:|---|---:|
| Indian News | 5,000 | English | 8 |
| Reuters News | 5,000 | English | 13 |
| Article News | 5,000 | English | 10 |
| Financial News | 5,000 | English | 8 |
| Variation News | 5,000 | English | 12 |
| Microsoft News | 5,000 | English | 8 |
| Thai News | 2,496 | Thai | 14 |

Sources in the manuscript are described as Kaggle for five datasets, Microsoft Research MIND for Microsoft News, and a private digital television provider for Thai News. Exact dataset URLs, versions, and access conditions must be documented by the authors; see [`data/README.md`](data/README.md).

## Outputs and reproducibility

Output CSV files contain run-level metrics and, for streaming, batch-level diagnostics. Each output directory also receives a JSON settings record. Preserve these outputs and the exact input dataset version used for any reported result. See [`docs/reproducibility.md`](docs/reproducibility.md) before preparing a reviewer response.

## License

No reuse license is included yet. Add a license only after confirming the authors' ownership and intended reuse terms. Dataset terms remain separate from the software license.
