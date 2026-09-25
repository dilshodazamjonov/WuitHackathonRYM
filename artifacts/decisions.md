# Decision log

| When | Decision | Evidence | Effect on CV AUC |
|---|---|---|---|
| Wed 24 Sep | Data layout is `data/training`, `data/test`, `data/sample_submission.csv` (renamed from the browser download name); `DATA_DIR` env var overrides | 0.2 | – |
| Wed 24 Sep | `DATE_FMT = %Y-%m-%d`, strict parsing; all 20,000 alert dates are 10-char ISO strings | 1, 2.1 | – |
| Wed 24 Sep | Timestamps are naive with a time part; keep as is, no timezone conversion. Hours 0–22 and weekdays are flat, so no hour-of-day or weekday features | F00a | – |
| Wed 24 Sep | No duplicate handling: 0 exact duplicate rows. Same-second pairs (1.3% of rows) sit almost entirely inside the pre-alert cluster; a gap of 0 is information, keep all rows | F00, 2.5 | – |
| Wed 24 Sep | No transactions after the alert date in train or test (0 rows with lag < 0), so the strict-pre-alert vs all question is moot. Lookback is a fixed 180 days | 2.5, F00b | – |
| Wed 24 Sep | Pre-alert cluster: about 8% of rows (median 31 per alert) fall in the last 3 minutes before the alert midnight; 152 alerts have none; 21 alerts carry it late on the alert day itself. Window `lag_days <= 1` covers all cases | 2.5, F00b | tbd |
| Wed 24 Sep | Window candidates for Phase 4: cluster (`lag_days <= 1`), 7, 30, 90, 180 days, plus recent-vs-baseline ratios; background activity declines from ~3 per day at 120 days to ~1 per day in the last week | F00b | tbd |
| Wed 24 Sep | `VAL = raw`: the index is one global scale (not per-alert standardized), near-normal with a right tail, floor at −2.91, type- and direction-dependent means; test `exp(index)` for money-like sums in Phase 4 | 2.6, F00c | tbd |
| Wed 24 Sep | Amounts are capped per type: the only 4 exactly repeated values are the per-type maxima (card 4.183, cash 4.863, international 6.430, transfer 6.692). Hit-the-cap flag is a Phase 4 candidate; 120 alerts hit a cap, escalation 13% vs 17% base, within noise | 2.6 | tbd |
| Wed 24 Sep | IDs, file row order, transaction-file position and alert date carry no signal (AUC 0.498–0.508); never used as features anyway | 2.7 | – |
| Wed 24 Sep | No shared histories: 0 transaction rows appear under more than one alert, so no pseudo-customers, no group CV, no linkage features | 2.8 | – |
| Wed 24 Sep | CV plan to confirm in Phase 2: stratified K-fold. Test IDs and dates are interleaved with train (30% test share in every id decile and every quarter); 2,405 positives allow 5 folds | 2.7 | – |
