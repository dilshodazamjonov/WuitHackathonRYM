# Phase 1 Reviewer Notes — Required Changes Before Phase 2

## Overall conclusion

Phase 1 is complete enough to move forward.

Nothing in Phase 0 or Phase 1 needs to be redone.

The overall project plan remains valid, but several branches of the original TODO can now be simplified, removed, or reinterpreted based on the actual data.

A key terminology change is recommended:

> Use **trigger window** or **pre-alert trigger burst** instead of “cluster.”

This is not a clustering algorithm. It simply refers to the dense block of transactions immediately before the alert.

---

## 1. Validation strategy

### Recommended split

Use a two-level validation design:

```text
14,000 labeled alerts
│
├── 80% development set
│   └── fixed 5-fold StratifiedKFold
│       ├── feature engineering decisions
│       ├── model comparison
│       ├── blend selection
│       └── OOF predictions
│
└── 20% confirmation holdout
    └── evaluated once after the solution is frozen
```

The 20% holdout should be created with a stratified split so the escalation rate remains representative.

### Development set

All iterative modeling decisions should be made using only the 80% development set.

Inside the development set:

- use one fixed 5-fold StratifiedKFold;
- generate OOF predictions;
- compare feature families;
- compare LightGBM and CatBoost;
- choose blend weights;
- perform error analysis;
- tune only against development CV.

The same folds should be reused across experiments so AUC differences are comparable.

### Confirmation holdout

The 20% holdout should not be inspected during feature or model development.

Do not repeatedly evaluate candidate models on it.

Use it once after the following are frozen:

- feature set;
- preprocessing;
- model families;
- hyperparameters;
- blend method and weights.

The holdout is therefore a **model-development confirmation set**, not a perfectly untouched dataset from the beginning of the competition, because some full-training label summaries were already inspected during Phase 1.

It is still valuable because future feature and model choices will not be optimized against its performance.

### How to interpret it

The main comparison should be:

```text
development OOF AUC
vs
confirmation holdout AUC
```

A small difference supports the reliability of the CV process.

A large drop on the holdout is evidence that the development process has overfit feature engineering, model selection, or blend selection.

The holdout should be used as a warning mechanism, not as another tuning set.

### After the holdout check

If the frozen solution behaves consistently on the holdout, reintroduce all 14,000 labeled alerts for final training.

The final hidden-test prediction should use the already frozen configuration on the complete training set.

A suitable final setup is:

```text
all 14,000 labeled alerts
        ↓
frozen feature pipeline
        ↓
5-fold training on the full dataset
        ↓
average hidden-test predictions across folds
        ↓
final submission
```

Do not keep 20% of the labeled data permanently excluded from the final competition model.

### Additional validation notes

- Stratified K-fold remains the primary CV method.
- Five folds are appropriate.
- Run adversarial train-vs-test validation in Phase 2 as a final confirmation that the hidden set resembles the training distribution.
- Time-based validation is optional as a stress test, not the primary selection metric.
- Group-aware validation is unnecessary because no shared alert histories were found.
- Additional CV seeds should be used only after the solution is largely frozen, not during every feature experiment.

---

## 2. Treat the pre-alert trigger window as a first-class analytical concept

The strongest structural finding is the dense transaction burst immediately before the alert.

The modeling logic should explicitly separate:

**historical baseline → trigger window → change between the two**

The trigger window should be defined explicitly instead of relying only on generic 1-day, 3-day, 7-day, 30-day, and 90-day windows.

A practical working definition is:

```python
trigger = lag_days <= 1
background = lag_days >= 2
```

This definition comes from the actual transaction timing structure observed in Phase 1.

The exact boundary may still be refined later if Phase 3 suggests a better cutoff.

### Important interpretation

The trigger burst itself should **not** be assumed to predict escalation.

Early class checks already suggest that trigger size alone is weak.

Therefore, the trigger window is primarily a **structural decomposition of the data**, not yet a proven predictive feature family.

The reason to isolate it is to test whether the behavior inside the trigger differs meaningfully from the alert's own historical baseline.

---

## 3. Trigger-window features to investigate

The first group of candidate features describes what happened immediately before the alert.

Examples:

```text
trigger_count
trigger_mean_amount
trigger_sum_amount
trigger_max_amount
trigger_cash_share
trigger_international_share
trigger_outgoing_share
trigger_n_types
```

These answer:

> What happened inside the trigger period?

They should not automatically be kept.

Phase 3 should test whether these quantities differ between dismissed and escalated alerts.

If they show little or no target separation, they should be dropped or treated only as ingredients for relative features.

---

## 4. Trigger-vs-background features are the stronger hypothesis

The more important hypothesis is that escalation may depend on **how abnormal the trigger behavior is relative to the alert's own history**.

Examples:

```text
trigger_count / background_daily_rate
trigger_cash_share - background_cash_share
trigger_international_share - background_international_share
trigger_outgoing_share - background_outgoing_share
trigger_mean_amount - background_mean_amount
trigger_amount_std - background_amount_std
```

These answer questions such as:

- Did cash usage suddenly increase?
- Did outgoing activity increase sharply?
- Did international activity appear when it was previously rare?
- Did transaction amounts shift relative to the customer's own historical pattern?
- Did activity become much more intense than usual?

This is currently a **hypothesis motivated by the Phase-1 structure**, not a confirmed target relationship.

It should be tested directly in Phase 3.

---

## 5. Add short-timescale trigger-burst features

Same-second activity should no longer be treated only as an audit curiosity.

The final pre-alert period contains a strong concentration of transactions occurring at identical timestamps.

This motivates candidate features such as:

```text
trigger_same_second_share
trigger_unique_timestamp_ratio
trigger_max_tx_per_second
trigger_duration_seconds
trigger_tx_per_minute
trigger_median_gap_seconds
trigger_min_gap_seconds
```

These measure transaction compression and velocity.

Two alerts can have the same trigger count but very different behavior:

```text
Alert A:
40 transactions spread across many hours

Alert B:
40 transactions compressed into 90 seconds
```

The count is identical, but the timing pattern is very different.

### Important caution

Phase 1 has **not yet shown that escalated alerts have more compressed trigger bursts**.

Therefore these features should be tested, not assumed to be useful.

If class distributions or CV show no difference, remove them.

---

## 6. Remove ordinary hour-of-day and weekday features

Generic hour-of-day and weekday behavior should be removed from the main modeling plan.

The data does not show realistic daily or weekly behavioral rhythms.

The unusual hour-23 concentration is explained by the pre-alert trigger burst rather than ordinary customer behavior.

Therefore remove or disable features such as:

```text
night_share
weekend_share
hour_of_day
weekday_of_transaction
```

The timing signal should instead focus on:

- time relative to the alert;
- trigger duration;
- transaction density;
- inter-transaction gaps;
- burst compression.

If an hour × weekday visualization was planned, replace it with a visualization of trigger-window timing or compression.

---

## 7. Post-alert handling is resolved

There is no meaningful post-alert transaction history.

The original branch comparing:

- strict pre-alert features;
- all supplied transactions;

is no longer necessary.

Use the supplied historical transactions directly while preserving the documented relative-time logic.

The small off-by-one behavior around the trigger boundary should remain documented, but it does not require two separate modeling pipelines.

---

## 8. Amount engineering should become channel-relative

Keep the raw standardized amount index.

It contains meaningful differences between alerts and between transaction channels.

However, global amount thresholds should not be the main approach.

Different transaction types occupy very different parts of the amount-index distribution.

Use channel-relative thresholds instead.

Preferred examples:

```text
card_in_above_channel_p95
card_out_above_channel_p95
cash_in_above_channel_p95
cash_out_above_channel_p95
transfer_in_above_channel_p95
transfer_out_above_channel_p95
international_in_above_channel_p95
international_out_above_channel_p95
```

The same principle can be applied to p99 thresholds or channel-standardized deviations.

Also prioritize:

- trigger-vs-background amount mean;
- trigger-vs-background amount variance;
- trigger-vs-background upper-tail share;
- amount changes within the same direction × type.

---

## 9. Treat `exp(index)` as experimental only

Do not describe `exp(index)` as restoring money values.

The observed distribution is compatible with a transformed log amount, but the exact original transformation is unknown.

It is acceptable to test:

```python
np.exp(miqdor_indeksi)
```

as an additional nonlinear representation.

Keep it only if cross-validation shows a meaningful improvement.

Raw-index statistics should remain the primary interpretable amount features.

---

## 10. Remove exact repeated-amount and round-amount features

The data does not support a meaningful round-amount or exact repeated-value pattern.

Remove or strongly deprioritize:

```text
max_same_amount_count
globally_frequent_amount_share
round_amount_flag
exact_repeat_amount_count
```

The few repeated exact values appear to come from channel-specific caps rather than real transaction behavior.

Structuring-style features can still use:

- low amount dispersion;
- many similar cash-in amounts;
- repeated transactions within short periods;
- coefficient of variation;
- narrow amount ranges.

The emphasis should be on **similar amounts**, not identical amounts.

---

## 11. Remove linkage and pseudo-customer logic

Delete the shared-history branch from the working plan.

Remove:

- pseudo-customer construction;
- transaction-hash linkage;
- group-aware CV;
- linked-alert features;
- linked-label features;
- reconstructed prior-alert behavior.

Every alert should be treated as an independent modeling unit.

The linkage audit can remain in the notebook as a documented integrity check, but it should not remain an active modeling branch.

---

## 12. Remove constant audit-derived model features

Some defensive variables are useful for data validation but have no modeling value because they do not vary.

Do not feed variables equivalent to these into the final model:

```text
has_history
n_post_signal_tx
duplicate_transaction_count
```

The code handling these edge cases can remain for robustness.

The distinction should be:

- keep the integrity checks;
- remove constant outputs from the model matrix.

---

## 13. Refine the date-feature decision

Keep the decision that these should never be model features:

```text
signal_id
row_order
raw_signal_date
```

However, do not treat all date-derived context as equivalent to raw date leakage.

A variable such as:

```text
alerts_same_day
```

is a workload or system-context feature rather than a raw temporal identifier.

It can still be tested in Phase 2.

Month, holiday, and other calendar variables should remain low priority unless the Phase-2 analysis shows a convincing relationship.

---

## 14. Interpretation of the early class differences

The early class comparisons suggest that raw trigger-burst size is not the main target signal.

The escalated and dismissed groups show only small differences in:

- trigger size;
- basic trigger type mix;
- trigger outgoing share.

Therefore, do not over-focus on simple volume.

The stronger modeling candidates are:

- trigger-vs-background ratios;
- changes in channel composition;
- outgoing-vs-incoming balance;
- amount behavior relative to the same channel;
- in-to-out timing;
- transaction velocity;
- short-timescale concentration;
- interactions between these behaviors.

The problem appears more likely to depend on **behavioral context** than on one raw transaction-count feature.

---

## 15. What Phase 3 should actually test about the trigger window

Before building a large trigger-feature family, Phase 3 should test a compact group of candidate variables.

Recommended first checks:

```text
trigger_count
trigger_count / background_daily_rate

trigger_outgoing_share
trigger_outgoing_share - background_outgoing_share

trigger_cash_share
trigger_cash_share - background_cash_share

trigger_international_share
trigger_international_share - background_international_share

trigger_mean_amount
trigger_mean_amount - background_mean_amount

trigger_same_second_share
trigger_tx_per_minute
trigger_median_gap_seconds
```

For each variable, examine:

- class distributions;
- escalation rate by decile;
- univariate AUC;
- missingness;
- stability across folds if used in a model.

The purpose is to determine whether the trigger window itself is predictive, whether the **change from baseline** is predictive, or whether neither adds useful information.

Do not keep the whole feature family merely because the trigger burst is visually striking.

---

## 16. Phase 2 validation objective

Phase 2 should now confirm rather than rediscover the train/test structure.

The key validation tasks are:

- confirm target distribution;
- plot escalation rate over time;
- confirm train/test weekly overlap;
- run adversarial train-vs-test classification;
- test alerts-per-day workload;
- create one stratified 80/20 development-holdout split;
- freeze one 5-fold StratifiedKFold inside the 80% development set;
- establish the first LightGBM baseline using development CV;
- keep the 20% confirmation holdout closed during subsequent feature and model development;
- create and validate a legal submission file from the current baseline.

The holdout should only be scored after the feature set, models, and blend are frozen.

Once the adversarial check and split are complete, validation design should be treated as frozen.

---

## 17. Priority order entering later phases

### P0

- explicit trigger-window features;
- trigger-vs-background ratios and deltas;
- direction × transaction-type aggregates;
- channel-relative amount behavior;
- outgoing/incoming flow behavior;
- pass-through timing;
- recent activity relative to historical baseline;
- standard window aggregates;
- LightGBM and CatBoost.

### P1

- same-second concentration;
- trigger-burst compression;
- inter-transaction gap behavior;
- dormancy and burst features;
- transaction sequences;
- selected interactions.

### P2

- workload context;
- cap-hit indicators;
- calendar context;
- `exp(index)` aggregates;
- other speculative nonlinear amount transformations.

---

## Final reviewer conclusion

The audit has reduced the search space rather than creating new uncertainty.

The main conclusions before Phase 2 are:

1. Use an 80% development / 20% confirmation-holdout split, with stratified 5-fold CV inside the development set.
2. Shared-history and group-CV logic can be removed.
3. Generic hourly and weekday behavior can be removed.
4. Exact repeated-amount logic can be removed.
5. The explicit pre-alert **trigger window** should become a core analytical unit.
6. The trigger burst itself is **not yet proven to predict escalation**.
7. The stronger hypothesis is that escalation depends on how the trigger behavior differs from the alert's own historical baseline.
8. Trigger-burst compression and same-second concentration are hypotheses to test, not confirmed signals.
9. Amount behavior should primarily be evaluated relative to transaction channel.
10. Score the confirmation holdout only once after the solution is frozen, then retrain the frozen pipeline using all 14,000 labels for the final submission.
11. Phase 2 can start without revisiting Phase 0 or Phase 1.
