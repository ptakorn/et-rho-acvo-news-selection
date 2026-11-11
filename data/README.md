# Data

Do not commit the article datasets here unless their licenses and the authors' permissions explicitly allow redistribution. The public repository should contain acquisition instructions or dataset source links instead of restricted raw files.

## CSV input format

The scripts accept UTF-8 CSV files. The minimum required column contains one news title per row. The column name is supplied with `--text-column`. An optional timestamp column can be supplied for chronological streaming via `--timestamp-column`.

If human labels are available, keep them in a separate column and do not use them for pseudo-label generation, TPE selection, or retraining. The current scripts do not treat human labels as training targets; independent labeled evaluation should be implemented as a separate, prespecified analysis.

## Data-source record to complete

The manuscript identifies the sources at a broad level only. Before public release, add the exact dataset name, URL, version/date accessed, license, number of rows retained, language, title field, any filtering/deduplication, and whether publication dates are available. For the Thai News dataset, document the provider and the permission basis without exposing confidential information.
