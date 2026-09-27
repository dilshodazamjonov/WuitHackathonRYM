# Alert escalation model — WIUT Hackathon 2026, FinTech track

Predict the probability that an automated financial-monitoring alert (`signal_id`) is escalated, using the transactions observed in the alert's 180-day lookback window. Metric: ROC-AUC on the hidden test set. One notebook holds the EDA, the model and the submission.

## Result

| | Alerts | ROC-AUC | Gini |
|---|---|---|---|
| Development set, 5 folds × 3 seeds, pooled out of fold | 11,200 | 0.645 | 0.290 |
| Holdout, scored once after the model was frozen (95% CI 0.626–0.678) | 2,800 | 0.651 | 0.301 |
| Final fit on all labelled alerts, pooled out of fold | 14,000 | 0.643 | 0.286 |

**Model:** LightGBM on 64 features, with 7 leaves, at least 200 alerts per leaf, feature fraction 0.7, learning rate 0.01 and early stopping inside each fold.
- The features are 28 full-history totals, 26 channel-relative amounts, 8 background channel shares and two amount contrasts.
- The test score is the mean of 15 models (5 folds × seeds 42, 7, 2026) trained on all 14,000 labelled alerts.
- The main signal is bank transfers that are small, and cash deposits that are large, for the alert's own card spending.

Every decision and its evidence is logged in `artifacts/decisions.md`.

## Layout

```
automated-alerts/
├── data/                       # competition files, gitignored
│   ├── training/               # train_signals.csv, train_transactions.parquet
│   ├── test/                   # test_signals.csv, test_transactions.parquet
│   └── sample_submission.csv
├── notebooks/solution.ipynb    # the one reproducible notebook
├── figures/                    # F00a–F17 PNGs exported for the EDA website
├── artifacts/                  # metrics, CV log, feature screen, ablation, split, insights, decisions.md
├── submissions/                # team_<TEAM_ID>.csv
├── frontend/                   # the EDA website
├── phase_0_4_summary.md        # team summary of all phases
├── requirements.txt            # pinned, exported from uv.lock
├── pyproject.toml, uv.lock
└── todo.md                     # EDA -> model plan
```

## Run

1. Use Python 3.13. Install the dependencies with `uv sync` or `pip install -r requirements.txt`.
2. The notebook also needs a Jupyter kernel in the same environment, and `ipykernel` is not pinned. Install it with `pip install ipykernel nbconvert`, or `uv pip install ipykernel` after `uv sync`.
3. Put the competition files under `data/` as shown above, or set `DATA_DIR=/path/to/data`.
4. Open `notebooks/solution.ipynb` and Restart & Run All, or run
   `jupyter nbconvert --to notebook --execute --inplace notebooks/solution.ipynb`.

**Runtime:** about six minutes on a laptop CPU (321–352 s on 8 cores, with LightGBM on 4 threads), peak memory about 3.1 GB. No internet is needed.

## Outputs

- `submissions/team_486052EC.csv`: the submission, with one row per test alert (6,000) and columns `signal_id` and `ehtimollik`. Its md5 is `6b50d37886f0c1b95f93c3f471e337e0`.
- `artifacts/metrics.json`: development, holdout and final-fit metrics, the submission check and the frozen configuration.
- `artifacts/cv_log.csv`: every cross-validation run with its fold scores.
- `artifacts/feature_screen.csv`: univariate AUC, fold range and IV of all 201 candidate features (the 199 from the feature builder plus the two amount contrasts).
- `artifacts/ablation.csv`: the family ablation.
- `artifacts/split.csv`: the frozen development/holdout split and folds. It is internal; do not publish it.
- `artifacts/insights.json`: the finding behind each figure.
- `figures/F00a`–`F17`: 20 figures for the website.

## Versions

Python 3.13.15, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0, scipy 1.18.1, matplotlib 3.11.2, pyarrow 25.0.1, and catboost 1.2.10 (used only for a comparison; the final model is LightGBM alone). All are pinned in `requirements.txt` and `uv.lock`.

Seeds are fixed, LightGBM runs deterministically on 4 threads, and every path is derived from the notebook location. Two clean runs therefore produce identical md5 hashes for every figure, artifact and the submission file. Other library versions can move the scores in the last digits.
