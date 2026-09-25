# Alert escalation model — WIUT Hackathon 2026, FinTech track

Predict the probability that an automated financial-monitoring alert (`signal_id`) is escalated, using the alert's 180-day transaction history. Metric: ROC-AUC on the hidden test set. One notebook holds the EDA, the model and the submission.

## Layout

```
automated-alerts/
├── data/                       # competition files, gitignored
│   ├── training/               # train_signals.csv, train_transactions.parquet
│   ├── test/                   # test_signals.csv, test_transactions.parquet
│   └── sample_submission.csv
├── notebooks/solution.ipynb    # the one reproducible notebook
├── figures/                    # F00... PNGs exported for the EDA website
├── artifacts/                  # F00 overview table, insights.json, decisions.md
├── submissions/                # team_<TEAM_ID>.csv
├── requirements.txt            # pinned, exported from uv.lock
└── todo.md                     # EDA -> model plan
```

## Run

1. Python 3.13. Install dependencies with `uv sync` or `pip install -r requirements.txt`.
2. Put the competition files under `data/` as shown above, or set `DATA_DIR=/path/to/data`.
3. Open `notebooks/solution.ipynb` and Restart & Run All, or run
   `jupyter nbconvert --to notebook --execute --inplace notebooks/solution.ipynb`.

Seeds are fixed and every path is derived from the notebook location, so two clean runs produce identical figures, artifacts and submission file.
