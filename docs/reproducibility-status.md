# Reproducibility status

This repository now contains a runnable reference implementation of the proposed ET-RHO and ACVO workflows. It was generated from the manuscript description; it has not been checked against the authors' original experiment code or archived outputs. Treat it as a reference implementation until those checks are completed.

| Item | Status |
|---|---|
| ET-RHO and ACVO source modules | Included under `src/et_rho_acvo/` |
| Static and streaming run commands | Included under `scripts/` |
| Seven dataset names, sample counts, language, and K | Transcribed from the manuscript into `configs/reproduction.yaml` |
| Exact dataset URLs, versions, filtering, and licenses | Still need to be added by the authors |
| Runtime/library versions from the original thesis | Python 3.7.6 and scikit-learn 0.24.2 are reported; this package targets Python 3.10+ and current APIs |
| Complete original hyperparameters | Not all provided; operational defaults are documented in the config and method notes |
| Reproduction of all baseline tables | Not included; ET-RHO/ACVO are implemented, but every baseline and transformer checkpoint is not |
| Exact reproduction of reported scores | Not established; reconcile seeds/settings and compare with original experiment archives |
| Independent semantic accuracy | Not available unless separately annotated ground-truth labels are supplied and held out |
| Software license | Not selected; confirm ownership and reuse intent before adding one |

## Before citing this repository in a reviewer response

- Review the code against the original implementation and experimental logs.
- Replace broad dataset source descriptions with exact source links, versions, licenses, and preprocessing steps.
- Reconcile the two seed ranges (1–30 versus 1000–1029) and the unspecified averaging convention and model parameters.
- Regenerate every reported table and figure from the released code, or state clearly which results are not reproduced.
- Do not upload raw datasets unless the applicable terms permit public redistribution.
- Add a software license only after confirming code ownership and desired reuse terms.
