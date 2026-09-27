# Phases 0–6 summary: data, validation, baseline, behavioral EDA, features, the final model and the submission

**Team RYM** · WIUT Hackathon 2026, FinTech track · status as of Sun 27 Sep 2026

All phases are done:
- 0 (setup)
- 1 (data audit)
- 2 (target, split and validation)
- 3 (behavioral EDA)
- 4 (EDA-driven features)
- 5 (feature selection and model)
- 6 (the one holdout check, the final fit and the submission)

Every number below comes from `notebooks/solution.ipynb` or the artifacts it writes. The notebook runs top to bottom in about six minutes and gives identical outputs on every run. Two markers flag numbers from elsewhere:
- *Earlier version*: from the longer notebook of Sat 26 Sep (commit `7628b55`). These checks were cut when the notebook was shortened on Sunday, and their results are logged in `artifacts/decisions.md`.
- *Side check*: from quick checks outside the notebook.

## In short

- **The data is clean.** It has no nulls, no duplicate rows, no orphan transactions and no alerts without a history, so no cleaning step is needed.
- **The test set is a random 30% of alerts from the same two years.** It is not a later period. Test and train cannot be told apart (adversarial AUC 0.497 on the totals, 0.500 on all 199 features), so ordinary stratified cross-validation estimates the hidden score.
- **Escalation is stable.** 17.2% of alerts are escalated, and the rate does not drift over 24 months. Workload, weekday, month-end and public holidays have no effect (*earlier version*).
- **Each history has the same shape.** Transactions are observed within a common 180-day lookback window: slowly declining background activity, then a dense burst in the last minutes before the alert.
- **The burst is structure, not signal.** Phase 3 tested it as the main behavioral hypothesis: neither the burst nor its change from the alert's background separates the classes on its own (13 trigger variables at AUC 0.470–0.528). Phase 4 confirmed it inside the model: the trigger family adds −0.0001 AUC on top of the amounts.
- **The signal is in amounts measured on their own channel's scale.** Escalated alerts make bank transfers that are small for their own card spending (AUC 0.397). Phase 4 found the mirror image in cash: incoming cash that is large for the alert's card level goes with escalation (AUC 0.549).
- **Features did most of the work.** Phase 4's one-seed ablation kept 72 features at 0.637. Almost all of that gain comes from one family, the channel-relative amounts (+0.019). Six of the nine families tested add nothing and are dropped.
- **The final model: 64 features and a lightly tuned LightGBM.** Averaged over three seeds, it scores 0.647 mean fold AUC and 0.645 pooled out of fold on the development set (Gini 0.290), against 0.614 and 0.612 for the baseline.
  - Phase 5 rechecked the features with three seeds and dropped compression, whose one-seed gain did not hold.
  - Pruning removed five rare-channel features.
  - Shallow trees (7 leaves) beat the defaults under every seed.
  - CatBoost and a blend added nothing, so the model is LightGBM alone.
  - On Sunday a wider search found two sharper amount contrasts, which were added to the 62 features: +0.007 pooled, 5 of 5 folds up. This reopened the Saturday freeze, and the deviation is logged.
- **The holdout agrees with the development estimate.** Scored once, it gives AUC **0.651** (95% CI 0.626–0.678) against 0.645 on the development folds. That is above the pre-registered warning line of 0.620, so there is no warning.
- **In queue terms,** reviewing the top 20% of alerts by score catches 34% of escalations. The top tenth escalates at 29.9% and the bottom tenth at 9.7%, against 17.2% overall.
- **The model's confident errors are mirror images of the other class** (*earlier version*, on the 72-feature model). They are not a pattern a missing feature would catch.
- **The submission is final.** `submissions/team_486052EC.csv` comes from the 64-feature model refit on all 14,000 alerts, with a pooled out-of-fold AUC of 0.643. It passes every format check, and its md5 is `6b50d37886f0c1b95f93c3f471e337e0`.

## The data

| | Train | Test |
|---|---|---|
| Alerts | 14,000 (with labels) | 6,000 |
| Transactions | 6,987,663 | 3,027,575 |
| Alert dates | 1 Jan 2025 – 31 Dec 2026 | same range |
| Transactions per alert | median 461 | median 460.5 |
| Lookback window | a common 180 days before the alert date | same |

Each alert has one row in the signals table: its ID, alert date and, for train, the label `eskalatsiya` (1 = escalated, 0 = dismissed). Each transaction has a timestamp, a direction (incoming or outgoing), a type (card, bank transfer, cash, international) and `miqdor_indeksi`, an amount index on an unknown scale.

The submission has one row per test alert with the escalation probability. It is scored by ROC-AUC.

## What the audit found (Phase 1)

| Question | Finding | What we do |
|---|---|---|
| Is the data clean? | 0 nulls, 0 duplicate rows, 0 orphan transactions, 0 alerts without a history | No cleaning step |
| Do train and test look alike? | Category shares agree within 0.2 points and amount quantiles agree to two decimals | Same pipeline for both |
| Do IDs or file order leak the label? | ID number, file row order and alert date all score AUC 0.499–0.508; the test share is 28–32% in every ID decile | Never used as features |
| Do alerts share customers? | 0 of 10,015,238 transaction rows appear under more than one alert | No group-aware folds, no linkage features |
| Is anything recorded after the alert? | 0 transactions after the alert date; transactions are observed within a common 180-day lookback window (the earliest transaction sits 180 days out for the median alert) | No leakage filter needed |
| Is there a daily or weekly rhythm? | Hours 0–22 hold 3.8% each and weekdays 14.3% each. Hour 23 holds 11.7%, and that is the pre-alert burst | No hour or weekday features |
| Where is the activity? | Background falls from about 3 transactions a day (120 days out) to about 1 a day in the last week. Then about 8% of all rows arrive in a burst: median 31 per alert, 1.5 minutes before the alert midnight, 99% within 3 minutes | Treat the trigger window (`lag_days <= 1`) and the background (`lag_days >= 2`) separately |
| Do all alerts have a burst? | 152 alerts have none; 21 have it just after midnight on the alert day itself | The window is defined by day (`lag_days <= 1`), not by clock time |
| What is the amount index? | One global scale, near-normal with a right tail and a floor at −2.9, not standardized per alert (per-alert means spread with sd 0.49). Each channel is its own bell curve: card lowest, international about 2 sd higher, outgoing above incoming | Amounts are compared within their own channel (direction × type), not on one global threshold. `exp(index)` is only an experiment |
| Are there round or repeated amounts? | Only 4 values repeat exactly, and each is the cap of one transaction type. 120 alerts hit a cap, escalating at 13% against 17% overall, within noise | No round-amount features; a cap flag is optional |
| Are same-second transactions errors? | 1.3% of rows share a second with another row of the same alert: 17% of burst rows but 0.04% elsewhere | Kept: they are part of the burst, and a gap of 0 seconds is information |

## Target, split and validation (Phase 2)

### Target

2,405 of 14,000 alerts are escalated, **17.2%**, about one for every 4.8 dismissed. AUC only ranks alerts, so no resampling is needed. The imbalance only means the split has to be stratified.

### The split, frozen

```
14,000 labeled alerts
├── Development set: 11,200 alerts (1,924 escalated)
│     5 fixed folds of 2,240 alerts, 384–385 escalated in each
│     Every feature, model and tuning decision is made here
└── Confirmation holdout: 2,800 alerts (481 escalated)
      Scored once, only after the whole model is frozen
```

- **Settings:** stratified on the label with seed 42, saved in `artifacts/split.csv`.
- **Holdout warning rule, fixed now before any scoring:** we get a warning if the development AUC beats the holdout AUC by at least 2 bootstrap standard errors (about 0.025).
- **What a warning means:** we check the development folds and prefer the simpler model, and we never tune on the holdout. The rule is a warning threshold, not proof that the model generalizes. In Phase 6 it was restated as a fixed number before the holdout was opened: a warning if the holdout AUC is below 0.620.
- **Final model:** the same frozen configuration, retrained with 5 folds on all 14,000 alerts. Test predictions are averaged over the folds and seeds.

### Is escalation stable, and is the test set like train?

| Check (development set only, where labels are used) | Result | Conclusion |
|---|---|---|
| Monthly escalation rate, 24 months | 13.6%–20.7%, no step or trend; chi-square p = 0.14; 2025 at 17.0% vs 2026 at 17.3% | Policy is stable, so no time-ordered validation |
| Alert volume vs rate (*earlier version*) | Volume moves in steps (up about 60% in July 2025, down in April and July 2026); alerts per day vs escalation gives AUC 0.49 | Busy days don't change outcomes; not a feature |
| Calendar (*earlier version*) | Weekday p = 0.45; last 3 days of the month p = 0.63; ±3 days around Navruz, Independence Day and both hayits p = 0.53 | No date or calendar features |
| Weekly test share | Within the 95% band of a random 30% draw in 101 of 105 weeks | Test is interleaved with train week by week |
| Adversarial validation (a model trained to tell test from train) | AUC 0.497, no feature above 5.4% of the importance | Test is the same population, so cross-validation on train is a fair estimate |

## Baseline model performance

**Features:** 29 totals over the whole 180-day lookback window. They are count, sum, mean, spread and maximum of the amount index, plus count, sum and maximum for each of the 8 direction × type channels. The burst and the background are not separated.

**Model:** LightGBM, trained and scored on the 5 development folds, with early stopping inside each fold.

| Metric | Value |
|---|---|
| **Development CV AUC** (mean of 5 folds) | **0.614 ± 0.016** |
| **Gini** (2 × AUC − 1) | **0.228** |
| Fold AUCs | 0.617 · 0.625 · 0.611 · 0.632 · 0.585 |
| Pooled out-of-fold AUC (all 5 folds' predictions together) | 0.612 |
| Trees kept by early stopping, per fold | 44 · 86 · 21 · 39 · 5 |
| Best single total on its own | 0.556 (largest outgoing bank transfer) |
| Test predictions (safety file, replaced by the final submission in Phase 6) | 6,000 rows, 0.118 to 0.324, median 0.165 |

**How to read it:** 0.5 is random and 1.0 is perfect. At 0.614 the model ranks a random escalated alert above a random dismissed one about 61% of the time. That is some signal, but weak.

**What the baseline already tells us:** early stopping after very few trees, plus a best single aggregate of only 0.556, means we are feature-limited rather than model-limited right now. Tuning the model would change little; the gain has to come from better features.

**Why it is weak, and where the headroom is:**

- **The totals dilute the one signal they contain.** The best baseline column is already a bank-transfer amount. Phase 3 shows why: the signal is transfer size relative to the alert's own card level, and summed totals per channel express that only indirectly. Separating the burst from the background, as we first expected, does not help by itself (see Phase 3).
- **The model ran out of signal fast.** At a learning rate of 0.03, early stopping ended after 5 to 86 trees in every fold, so the limit is the features, not the model settings.
- **No single total is strong.** The best one reaches AUC 0.556 on its own. That does not mean new features must each beat 0.614 alone: a feature with a univariate AUC of 0.52 can still add information the totals lack, which is why families are judged by how much they improve the model's OOF AUC.
- **The CV score is only slightly optimistic** (*side check*). Early stopping on the scored fold adds about 0.004 AUC, far below the 0.025 holdout warning threshold.

## Behavioral EDA (Phase 3)

### How we compared the classes

- **Development set only**, 11,200 alerts. The holdout never appears.
- **Three parts per history:**
  - the background (2 or more days before the alert date)
  - the trigger window (the alert day and the day before)
  - the burst core (the last hour)
- **One row per alert, not per transaction**, so long histories do not dominate. Comparisons use ECDFs and escalation rates by decile with 95% intervals.
- **Noise floor:** a single AUC has a standard error of 0.007 under chance, so anything between 0.486 and 0.514 cannot be told from noise. About 120 variables were screened, so about 6 would pass that band by luck. We only act on effects well outside it, or on effects that repeat across views.

### What the history looks like before any label

- **The burst has its own composition.** It is a rapid run of small card payments: 60% card against 53% in the background, 32% transfers against 40%, and a mean amount index of −0.50 against −0.10.
- **The trigger window and the burst differ in length** (*earlier version*). The burst spans a median 2.8 minutes, but the trigger window spans 135 minutes, because 7,200 alerts also have a few ordinary transactions earlier on the day before.
- **Edge cases** (*earlier version*):
  - 119 alerts have no background at all, so any change from the background is undefined for them.
  - 132 alerts have no burst in the last hour.
  - In 10 alerts the burst sits earlier on the day before.
- **Weekly transaction volume** follows how many alert windows cover each week. The channel mix is flat over time in train and test (card 51–58%, transfers 35–42% of a week).

### Results by hypothesis

| Hypothesis | Result on the development set | Verdict |
|---|---|---|
| The trigger burst, or its change from the background, drives escalation | 13 trigger variables at AUC 0.470–0.528; trigger size relative to background 0.494; all 13 together CV AUC 0.547 | **Not supported** on its own |
| Escalated alerts build up activity before the alert | Same curve shape for both classes; escalated about 5% higher at every lag (median ratio 1.055), no build-up | Level only, no special window |
| More background activity | Median 447 vs 415 transactions, AUC 0.527 | Weak |
| Faster or more concentrated bursts | AUC 0.480–0.514 | Not supported |
| Pass-through (money in, then quickly out) | Trigger outgoing share 0.528, outflows within 24 h of an inflow 0.526, shorter inflow-to-outflow time 0.478 | Right direction, small |
| Channel mix shifts before escalation | Every share and change 0.476–0.527 | Not supported |
| Risky transaction sequences | Common-pair lift 0.96–1.19, every transition within 0.03 of chance | Not supported |
| **Unusual amounts within a channel** | **Transfers smaller for escalated alerts; relative to own card level AUC 0.397** | **Supported: the main signal** |

### The main finding: small transfers for the alert's own spending level

- **Transfers are smaller.** Escalated alerts' mean outgoing transfer is 0.15 channel standard deviations smaller (AUC 0.426), and the mean incoming transfer 0.12 smaller (0.439). International transfers are smaller too. Card and cash differ by 0.06 or less.
- **It is a stable trait, not a change before the alert.** Escalated alerts' outgoing transfers are 0.10–0.13 standard deviations smaller in every band, from six months out to the trigger window itself.
- **It sharpens against the alert's own level.** An alert's amounts move together across channels (Spearman 0.47–0.87 between its channel means, *earlier version*). Outgoing transfer size minus outgoing card size scores AUC 0.397, with every fold between 0.375 and 0.430, while card size alone scores 0.495.
- **The interaction is visible directly.** In the middle card-size quintile, escalation falls from 25% for the smallest transfers to 4% for the largest. For mid-sized transfers, it rises from 13% with the smallest card payments to 24% with the largest.
- **Across deciles** of the relative measure, the escalation rate runs from about 24% down to 9%, about 2.5 times apart.
- **How we read it:** escalated alerts move money by bank transfer in smaller amounts than their own card spending would suggest. The pattern recalls structuring, but the data cannot say why analysts escalate it.
- **Confirmed in Phase 4:** adding the amounts family to the baseline lifts the development CV AUC from 0.614 to 0.633 (see below).

### Decisions logged

- **Trigger boundary** (*earlier version*): `lag_days <= 1` for counts, shares and amounts. Its AUCs are within 0.002 of the last-hour core, and it also catches the 10 early bursts. The last-hour core is used for timing.
- **Background windows:** no window is special, so 7, 30 and 90 days stay as default candidates only.
- **Error analysis** of the model's most confident misses: done in Phase 4, on the first behavioral model (*earlier version*).

## Features and ablation (Phase 4)

### How the features were built

- **One builder for train and test.** `build_features()` turns every history into one row with the same code for both sets.
- **No labels in the features.** Channel statistics (mean, sd, p95, p99 and the per-type cap for each direction × type) come from train and test transactions together.
- **The baseline stays in.** The 29 totals remain as the reference, and every other feature belongs to one family, so the ablation keeps or drops each family as a whole.
- **199 features in total.** None is constant or a copy of another, and missing values stay missing rather than imputed.
- **Left out on purpose:** alerts per day and the alert-date calendar, because Phase 2 showed they carry no signal (*earlier version*).

| Family | Features | What it holds |
|---|---|---|
| base | 29 | the baseline totals |
| amounts | 30 | per channel: background mean of the channel z-score, share above the channel's p95 and p99; per direction: transfer, international and cash size minus card size |
| windows | 36 | background counts over lags 2–7, 2–30, 2–90 and 2–180, in total and per channel |
| typology | 6 | outflows within 24 h of an inflow, hours from inflow to outflow, in/out ratio, spread of incoming cash amounts |
| trigger | 13 | the trigger-test variables without the timing ones; the rate of the last 7, 30 and 90 days against the whole background |
| mix | 8 | background share of each channel |
| compression | 5 | burst timing: same-second share, speed, duration, median gap, silence before the burst |
| rhythm | 5 | background active days, busiest day, gaps between transactions |
| sequence | 64 | transition shares between consecutive channels |
| context | 3 | transactions at a type's cap, `exp(index)` sums for the trigger and the background |

### Screening and drift

- **The strongest single features are all amounts.** The six strongest and ten of the top 20 come from the amounts family. Outgoing transfer minus outgoing card size leads at 0.397, with every fold between 0.375 and 0.430. Even its incoming counterpart (0.433) beats the best baseline total (0.441).
- **Every other family stays near chance.** Apart from the baseline totals, each family's strongest feature lies within 0.04 of chance; the trigger family's best is 0.030.
- **No leaks.** The highest information value is 0.13 and the strongest AUC 0.397 (0.603 in the other direction), far from anything that would suggest a leak.
- **No drift.** The highest PSI is 0.007 (below 0.10 counts as stable), and no feature reaches 0.10. The train-vs-test classifier stays at chance with all 199 features (AUC 0.500), and no feature takes more than 1.3% of its gain.

### Ablation: which families pay off

Families were added one at a time to the 29 totals, in the order the EDA ranked them, on the same five development folds with one seed. A family is kept when the mean fold AUC rises **and** at least 4 of 5 folds improve against the last kept set.

| Family added | Features added | CV AUC after | Change | Folds up | Kept |
|---|---|---|---|---|---|
| amounts | 30 | 0.6326 | **+0.0187** | 4 | **yes** |
| windows | 36 | 0.6312 | −0.0014 | 2 | no |
| typology | 6 | 0.6306 | −0.0020 | 1 | no |
| trigger | 13 | 0.6325 | −0.0001 | 2 | no |
| mix | 8 | 0.6348 | +0.0022 | 4 | yes |
| compression | 5 | 0.6369 | +0.0021 | 4 | yes |
| rhythm | 5 | 0.6325 | −0.0044 | 1 | no |
| sequence | 64 | 0.6334 | −0.0035 | 2 | no |
| context | 3 | 0.6333 | −0.0036 | 2 | no |

**Ablation-kept set:** the 29 totals plus amounts, mix and compression, 72 columns. Phase 5 rechecked it with three seeds and trimmed it to 62 (below).

| Metric | Baseline (29 totals) | Kept set (72 features) |
|---|---|---|
| **Development CV AUC** | 0.614 ± 0.016 | **0.637 ± 0.028** |
| **Gini** | 0.228 | **0.274** |
| Fold AUCs | 0.617 · 0.625 · 0.611 · 0.632 · 0.585 | 0.640 · 0.665 · 0.609 · 0.670 · 0.601 |

### What the kept model uses

Family shares are *earlier version*; the six features below are in F12.

| Family | Features | Share of LightGBM gain |
|---|---|---|
| amounts | 30 | 46.8% |
| base | 29 | 33.3% |
| mix | 8 | 12.2% |
| compression | 5 | 7.7% |

The six features with the most gain:

1. **Outgoing transfer minus outgoing card size** (9.8% of gain). Escalation falls from 24–25% in the two lowest deciles to 9% in the highest.
2. **Incoming cash minus incoming card size** (6.4%). Escalation rises from 12% to about 20% across the deciles.
3. **Incoming transfer minus incoming card size** (2.7%). It repeats the first pattern more weakly, about 22% down to 12%.
4. **Background outgoing transfer size** (2.5%). The same pattern again, 23% down to 12%.
5. **Largest incoming cash amount** (2.4%). Nearly flat on its own (AUC 0.512).
6. **Background share of incoming cash** (2.3%). Also nearly flat on its own (AUC 0.520).

The last two carry gain only through combinations with other features.

### What stood out

1. **Cash runs the other way from transfers.** This is the one new direction Phase 4 found. Escalated alerts send transfers that are *small* for their own card level, but deposit cash that is *large* for it (AUC 0.549). They also make more outgoing cash transactions (0.541). The model ranks this cash difference second of all 72 features.
2. **Passing the screen is not the same as adding information.** Twenty-one of the 36 window counts lie outside the chance band, yet the family lowers the model's AUC. The counts that separate the classes, such as outgoing cash (0.539 in the background count), are already in the totals (0.541), so the model gains nothing new.
3. **The burst adds nothing, even inside the model.** Phase 3 showed the trigger variables are not predictive on their own. Added to the model on top of the amounts, the trigger family changes the AUC by −0.0001 (2 of 5 folds up).
4. **Pass-through points the right way but does not help.** In Phase 3 the typology variables leaned the expected way (0.526–0.528). As a family they lower the model's AUC (−0.002, 1 of 5 folds up).
5. **Two families were kept by a hair.** Mix and compression each add +0.002. Phase 5 reran them with three seeds: mix holds, compression does not.
6. **International transfers are strong but mostly missing.** Outgoing international minus card size scores 0.434, one of the strongest features. But two thirds of alerts (67%) never use the channel, so it can help only a third of the queue.
7. **The fold spread grew.** The kept set's fold AUCs range from 0.601 to 0.670 (± 0.028), against ± 0.016 for the baseline. The amounts gain varies by fold, from −0.002 on the third fold to +0.034 on the second. The fifth fold stays the weakest, as it was for the baseline.
8. **No drift on any of the 199 features.** This supports the Phase 2 finding that the test set is the same population.

### Error analysis: the confident misses are mirror images (*earlier version*)

We read the 20 escalated alerts the kept model ranks lowest and the 20 dismissed alerts it ranks highest, out of fold on the development set.

| | 20 escalated ranked lowest | All escalated | All dismissed | 20 dismissed ranked highest |
|---|---|---|---|---|
| Model score (median) | 0.064 | 0.180 | 0.155 | 0.445 |
| Outgoing transfer minus card size | +0.30 | −0.28 | −0.14 | −0.48 |
| Incoming cash minus card size | −0.14 | +0.01 | −0.05 | +0.12 |
| Transactions in the history | 310 | 490 | 453 | 664 |
| Raw outgoing transfers, channel sd | +0.84 | −0.10 | +0.02 | −0.30 |

- **Missed escalations** have short histories in which everything is large: their transfers, and their card payments too (+0.28 outgoing, +0.40 incoming).
- **Confident false alarms** are long, busy histories whose transfers are far below their card level.
- **Each group is a stronger version of the other class** on every variable in the table. The model is not missing a pattern these alerts share.
- **How we read it:** what decided these alerts is probably information outside the transaction history, such as the rule that fired or the customer's profile, and neither is in this data.

### Side checks (outside the notebook)

- **Shrinkage does not help.** Shrinking the per-alert channel means towards the channel average for short histories (the missed escalations are short) scores 0.635 against 0.637: no effect.
- The seed-stability check that first stood here is now in the notebook, as the Phase 5 feature-set comparison below.

### Decisions

- **Kept by the ablation:** the 29 totals plus amounts, mix and compression, 72 columns, at a development CV AUC of 0.637 on one seed. This became the starting candidate for Phase 5.
- **Dropped:** windows, typology, trigger, rhythm, sequence and context. Each lowered the mean fold AUC.
- **Compression was provisional** because its gain was the size of seed noise. Phase 5 dropped it (below).

## Feature selection and the frozen model (Phase 5)

Everything here uses the development folds only. The holdout is still closed.

### Which feature set: four candidates under three seeds

The candidates were compared with the default LightGBM settings and seeds 42, 7 and 2026. A larger set replaces a smaller one only if it gains under every seed **and** at least 4 of 5 folds improve on the seed-averaged fold AUCs.

| Candidate | Features | Seed 42 | Seed 7 | Seed 2026 | Mean |
|---|---|---|---|---|---|
| amounts only | 30 | 0.6316 | 0.6308 | 0.6292 | 0.6306 |
| base + amounts | 59 | 0.6326 | 0.6321 | 0.6311 | 0.6320 |
| **base + amounts + mix** | **67** | 0.6348 | 0.6350 | 0.6356 | **0.6351** |
| + compression | 72 | 0.6369 | 0.6341 | 0.6310 | 0.6340 |

- **Base alone is not enough.** The 29 totals add only +0.001 to +0.002 to the amounts family and improve 3 of 5 folds.
- **Base with mix clears the rule easily.** Together they beat amounts alone under every seed (+0.003 to +0.006) and in 5 of 5 folds.
- **Compression was one-seed luck.** Its +0.002 on seed 42 turns into −0.001 and −0.005 under the other seeds (2 of 5 folds up), so it is dropped.
- **The amounts family alone (30 features) already reaches 0.631.**

### Pruning: 67 → 62

- **Importance is stable where it matters.** Over the 15 fits (5 folds × 3 seeds), the three strongest features take 8–19%, 6–10% and 2–5% of the gain in every fit. None of the top eight ever has a fit with zero gain.
- **Five features are dropped.** Each stays below 0.2% of the gain in all 15 fits. All describe rare events: three international tail shares, the top-1% share of outgoing cash, and the outgoing international count.
- **Almost no redundancy.** Only one pair of the 67 is correlated at |ρ| ≥ 0.95 (outgoing international count and sum), and its weaker half is among the five.
- **A borderline call, kept by team decision.** The pruned set scores 0.6334 against 0.6351. The unpruned set is ahead under two seeds but behind by 0.0001 under the third, so by the rule the smaller set is kept. The difference is within seed noise either way.

### The model

- **LightGBM, lightly tuned.**
  - The grid covered 7/15/31 leaves × 50/200 per leaf × feature fraction 0.5/0.7, on one seed. All four 7-leaf points land in the top five, and the defaults come last (0.6323), although the whole grid spans only 0.005.
  - The best point is 7 leaves, at least 200 alerts per leaf and feature fraction 0.7, with the learning rate lowered to 0.01.
  - It beats the defaults under every seed (+0.008, +0.006, +0.003) and in 4 of 5 folds: 0.639 averaged over the seeds, against 0.633.
- **CatBoost with default settings: 0.635** on seed 42. It trails LightGBM in 4 of 5 folds and ranks alerts much like it (out-of-fold Spearman 0.93). The two share 8 of their top 10 features and the same top two.
- **No blend.** Rank blends at LightGBM weights 0.3 / 0.5 / 0.7 reach a mean fold AUC of 0.638 / 0.639 / 0.640, against 0.640 for LightGBM alone. The best is ahead in only 2 of 5 folds and has a lower pooled AUC (0.6384 against 0.6387). The model is LightGBM alone.

### Two more amount contrasts (Sunday, a logged deviation)

The configuration was first frozen on Saturday with 62 features. On Sunday we ran a wider search on the development folds. It ran outside the notebook, with the holdout still closed: about 1,550 feature sets built around the channel-relative amounts, with permuted features as nulls. Two contrasts held up, and both are sharper versions of the main finding:

- **`outgoing_transfer_minus_card_z_p75`** takes the upper quartile of the alert's background outgoing-transfer size minus the upper quartile of its outgoing-card size. The existing feature compares means. It scores AUC 0.387 on its own, against 0.397 for the mean-based version.
- **`cash_minus_transfer_contrast_z`** is incoming cash minus card, less the average of the two transfer-minus-card contrasts. It is high when cash is large and transfers are small for the card level. It scores AUC 0.609 on its own, against 0.549 for incoming cash minus card.

| Tuned LightGBM on the development folds | 62 features | 64 features |
|---|---|---|
| Mean fold AUC, seeds 42 / 7 / 2026 | 0.6400 / 0.6384 / 0.6384 | 0.6467 / 0.6474 / 0.6464 |
| Gain per seed | – | +0.0068 / +0.0090 / +0.0080 |
| Folds up (seed-averaged) | – | 5 of 5 |
| Pooled out of fold | 0.6378 | 0.6452 |

- **It passes the rule every feature-set change had to pass:** a gain under every seed, and at least 4 of 5 folds up.
- **On three fresh splits of the development set, the 64 win every time.** This check was reported only, and it ran before the holdout.
  - Mean fold gain: +0.0067 / +0.0051 / +0.0101
  - Pooled gain: +0.0077 / +0.0017 / +0.0077
- **Part of the gain is likely selection inflation, about +0.001 to +0.002,** because the pair came out of a search on these same folds.
- **The search's own pick was not adopted.** That was a 53-feature set at +0.015 over the 62, but its extra gain over the pair could not be separated from selection noise.
- **Tuning, CatBoost and the blend were not reopened.**

### The frozen model on the development folds

| Metric | Baseline (29 totals, one seed) | Frozen model (64 features, 3 seeds) |
|---|---|---|
| **Mean fold AUC** | 0.614 ± 0.016 | **0.647 ± 0.028** |
| **Pooled out-of-fold AUC** | 0.612 | **0.645** |
| **Gini** (pooled) | – | **0.290** |
| KS | – | 0.241 |
| Fold AUCs | 0.617 · 0.625 · 0.611 · 0.632 · 0.585 | 0.646 · 0.679 · 0.615 · 0.678 · 0.617 |

- **The seeds barely disagree.** Their out-of-fold predictions correlate at 0.990–0.993, and their mean fold AUCs lie within 0.001 of each other (0.6464–0.6474).
- **The folds differ far more than the seeds.** Fold AUCs range from 0.615 to 0.679, so which alerts land in a fold matters more than the random seed.
- **No decay over time.** Trained on the oldest 80% of the development alerts and scored on the newest 20% (from 7 June 2026, 387 escalated), the model reaches 0.673 (SE ≈ 0.016). That is no lower than on the shuffled folds. The check is reported, not used to choose.

**What drives it** (SHAP out of fold, F17):
- **Where the weight is:** amounts carry 65% of the SHAP weight, totals 27% and channel shares 9%.
- **The two new contrasts rank first and second:**
  - the cash contrast holds 17.5% of the weight; a higher value raises the score (Spearman between value and SHAP +0.80)
  - the upper-quartile transfer contrast holds 15% (−0.89)
- **All ten strongest features push the score the same way as they separate the classes on their own:**
  - transfers that are small for the alert's card level raise it (−0.87 for outgoing transfer minus card)
  - incoming cash that is large for the card level raises it (+0.84)
  - a large single cash deposit raises it (+0.93)
- **A point a business reader may question:** larger amounts in general *lower* the score (the largest transaction −0.80). That is how alerts were decided in this data, where escalated alerts make smaller transfers. It does not say that large amounts are safe.

**Business metrics** (F16):

| Review the top … of the queue | Share of escalations caught | Escalation rate in that slice |
|---|---|---|
| 10% | 17.4% | 29.9% (1.7× the 17.2% base rate) |
| 20% | 34.4% | 29.5% |
| 30% | 48.5% | 27.8% |
| bottom 10% (for contrast) | – | 9.7% |

- **What the score is for:** it splits the queue into a riskier top and a calmer bottom. It does not isolate a group that is safe to close unseen, so use it to order the queue, not to drop alerts.

### What stood out in Phase 5

1. **The one-seed ablation overstated a family.** Compression looked like +0.002 on seed 42 and was −0.005 under seed 2026. Three seeds were enough to show it.
2. **Shallow trees win.** Seven leaves beat 31 leaves at every setting of the other two parameters. This fits a weak signal carried by a few smooth relations, such as transfer and cash size against the card level. Tuning adds about 0.006, small next to the +0.02 from features.
3. **A second model adds nothing.** CatBoost ranks alerts almost the same way (0.93) and a bit worse, and no blend weight helps.
4. **The newest alerts are scored no worse** (0.673 on the time split). There is no sign the pattern fades.
5. **One sign flips once the cash features are together** (*earlier version*, on the 62-feature model). The background share of incoming cash (SHAP rank 14) lowers the score at high values, although on its own it goes slightly with escalation (0.520). Its effect is most likely conditional on the other cash features, which carry the main cash signal.
6. **The pruning decision is a coin flip,** 0.6334 against 0.6351, within seed noise. We keep the smaller set by the rule.
7. **Features beat tuning again.** The two contrasts added +0.007 pooled, more than all of the tuning (+0.006).

### Frozen configuration (logged in `decisions.md`)

| | Setting |
|---|---|
| Features | 64: the 29 totals, amounts and mix, minus five rare-channel features, plus the two amount contrasts |
| Preprocessing | none; missing values left to LightGBM |
| Model | LightGBM alone, no blend |
| LightGBM | learning rate 0.01, 7 leaves, at least 200 per leaf, feature fraction 0.7, bagging fraction 0.8, lambda_l2 1.0, early stopping 200 rounds on the fold's AUC |
| Seeds | 42, 7, 2026, predictions averaged |
| Pooled dev OOF AUC | **0.645** |
| Holdout warning rule | warning if the holdout AUC is below **0.620**: the pooled dev AUC minus about 0.025, two bootstrap SE at 481 escalated alerts. The number was fixed before the holdout was opened; the bootstrap SE and CI are reported only. A warning threshold, not proof of generalization |

## Holdout check, final fit and submission (Phase 6)

The model is LightGBM alone, so the holdout check and the final fit use the notebook's `fit_seeds` (three-seed LightGBM). The two-model `fit_blend` in the `todo.md` appendix is not needed.

### The holdout, scored once

The frozen 64-feature model scored the 2,800 holdout alerts (481 escalated). It was trained on the five development folds × three seeds, exactly as in Phase 5. The warning line of 0.620 was fixed before the cell first ran.

| | ROC-AUC | Gini |
|---|---|---|
| Development folds, pooled out of fold | 0.645 | 0.290 |
| **Holdout** | **0.651** | **0.301** |
| Holdout bootstrap SE and 95% CI (1,000 resamples, reported only) | 0.013, 0.626–0.678 | |

- **No warning.** The holdout sits 0.005 above the development estimate and well above the 0.620 line. Nothing was changed after it.
- **What it shows:** the cross-validation process gave an honest estimate.
- **What it does not show:** it is not proof that the model generalizes. At SE 0.013 the holdout alone is too wide to confirm the +0.007 from the two contrasts.
- **Only this one evaluation is used for any decision.** The notebook recomputes the same number on every run, deterministically, so later runs are reproduction checks, not new looks. Only the 64-feature model was ever scored on the holdout.

### The final fit

- **Same recipe on all labels.** The frozen configuration was refit on five folds over all 14,000 labelled alerts (`final_fold` in `split.csv`), with early stopping inside each fold. The test score is the mean of the 15 models (5 folds × 3 seeds).
- **Pooled out-of-fold AUC on all 14,000: 0.643** (folds 0.632–0.671). That lies between the development 0.645 and the holdout 0.651, so the extra 2,800 labels do not change the picture.
- **Sanity checks:**
  - every test alert has a score, and none is missing
  - the seeds agree on test (Spearman 0.995–0.998)
  - test scores are centred like the out-of-fold ones (mean 0.172 against 0.171, median 0.153 against 0.157), with tighter tails (1st–99th percentile 0.097–0.313 against 0.071–0.346)
- **Why the tails are tighter** (*side check*):
  - Each model scores the test set with the same spread as its own held-out fold (sd 0.059 against 0.059 on average), so the test set is not a different population.
  - The cause is averaging. A test score is the mean of all 15 models, while an out-of-fold score comes from the three models of one fold, and averaging pulls the extremes in.
  - Drift was already ruled out in Phase 4 (PSI at most 0.007, adversarial AUC 0.500).
- **Three of the fifteen final models stop almost at once** (*side check*):
  - Early stopping ends them after 7–12 trees, on folds where the validation AUC is flat or peaks that early. The other 12 keep 134–1,601 trees.
  - None of the development-fold models behind the holdout score stopped below 256 trees.
  - Dropping the three barely moves the test ranking (Spearman 0.99996 with the full mean), so the frozen configuration stays.
  - A minimum tree count would be the first thing to test next time.

### The submission

- **The file:** `submissions/team_486052EC.csv`, 6,000 rows, one per test alert, with scores from 0.084 to 0.359 (median 0.153).
- **The check:** the validator passes on columns, row count, IDs, missing values, the 0–1 range and the file name. The md5 is `6b50d37886f0c1b95f93c3f471e337e0`.
- **Reproducible:** two clean full runs give the same md5 for this file and for every figure and artifact (29 files). A run takes 321–352 s and peaks at about 3.1 GB of memory.
- **The baseline safety file it replaces** is still written first on every run, and its md5 (`6d42902c…`) is printed in the notebook.

## What is left (Sun 27 Sep)

| Who | Task |
|---|---|
| QA | Reproduce on a second laptop from a fresh clone, with the data copied into `data/` (see the README; `ipykernel` must be installed). Expect 6,000 rows and md5 `6b50d378…`, or near-identical scores if library versions differ |
| Website | F16 and F17 now show the 64-feature model. Add the holdout result from `metrics.json`, then check that the public URL opens in an incognito window without a login |
| Submission | Confirm the TEAM_ID `486052EC` with the organisers, then submit `team_486052EC.csv`, the website URL and the notebook by 18:00 (deadline 23:59). Keep a screenshot of the confirmation |

## For the team

**Website (EDA site)**
- Figures ready in `figures/`, 20 in total:
  - Audit and validation:
    - F00a hour and weekday
    - F00b relative time and the burst
    - F00c amount index
    - F01 target and monthly rate
    - F02 alerts per week, train vs test
  - Behavioral EDA:
    - F03 background volume by class
    - F04 the trigger test, 13 decile curves (**key figure**)
    - F05 weekly transactions by type
    - F06 background activity before the alert by class
    - F07 incoming vs outgoing
    - F08 channel mix
    - F09 amounts by channel and class (**key figure**)
    - F10 burst compression
    - F11 transaction sequences
  - Features:
    - F12 the six features with the most gain, escalation rate by decile
    - F13 family ablation: which EDA ideas paid off (**key figure**)
    - F14 the 25 strongest single features
    - F15 drift: PSI and the train-vs-test classifier on all features
  - Model (updated Sunday for the 64-feature model):
    - F16 ROC, escalations caught when reviewing the top of the queue, escalation rate by score decile (**key figure**)
    - F17 what drives the model: gain share and SHAP values
- Each figure's finding and action are in `artifacts/insights.json`.
- Tables for the website:
  - `artifacts/feature_screen.csv`: every feature's AUC, fold range, IV and family
  - `artifacts/ablation.csv`: the family ablation
  - `artifacts/metrics.json`: the development metrics, the holdout result, the final fit, the submission check and the frozen configuration
- Suggested placement:
  - Target and behavior section: F04 and F09 go together (the hypothesis we tested and what the data showed instead). F03, F06, F07, F08, F10 and F11 fit the same section as supporting views.
  - Transaction history section: F05.
  - "Features and modeling ideas motivated by EDA" section: F13, then F12 and F14. F13 is the story in one chart: the amounts pay off, and the burst, windows and sequences do not.
  - Validation section: F15, next to F02.
  - Conclusion section: F16, with the queue numbers (top 20% catches 34% of escalations), and F17 for how the score can be explained to analysts. Quote the holdout AUC (0.651, 95% CI 0.626–0.678) as the one clean estimate. Quote the development AUC and the queue numbers as development out-of-fold numbers.
- Publish only aggregated figures and numbers: never raw data, never `split.csv`.

**QA and submission**
- `submissions/team_486052EC.csv` is the final submission: the 64-feature model refit on all 14,000 alerts, md5 `6b50d37886f0c1b95f93c3f471e337e0`.
- The notebook's validator checks columns, row count, IDs, missing values, the 0–1 range and the file name.
- A full run takes about six minutes, and two consecutive runs give identical md5 for every figure, artifact and the submission.
- Running the notebook from the command line needs `ipykernel` in the same environment. It is not in `requirements.txt`; the README shows how to add it.
- **Open item:** confirm with the organisers that `486052EC` is our TEAM_ID before we submit.

**Files produced**

| File | What it is |
|---|---|
| `notebooks/solution.ipynb` | The notebook: Setup, Load, Data audit, Target / split / validation, Behavioural EDA, Feature engineering, Screening / drift / ablation, Feature selection, Models on the development folds, Frozen model, Holdout check, Final fit, Submission, Exports |
| `README.md` | How to run, versions, runtime and outputs |
| `artifacts/F00_dataset_overview.csv`, `F00_schema.csv` | Dataset overview and column descriptions |
| `artifacts/split.csv` | Frozen split and folds (internal only) |
| `artifacts/cv_log.csv` | Every model run with its fold scores: ablation steps, feature-set candidates per seed, tuning grid, CatBoost, the frozen model |
| `artifacts/feature_screen.csv` | Univariate screen of all 199 features |
| `artifacts/ablation.csv` | Family ablation table |
| `artifacts/metrics.json` | Development, holdout and final-fit metrics, the submission check and the frozen configuration |
| `artifacts/insights.json` | Figure findings for the website |
| `artifacts/decisions.md` | Every decision with its evidence, including the frozen configuration, the holdout rule and result, and the final fit |
| `figures/F00a`–`F17` | Twenty figures (PNG) |
| `submissions/team_486052EC.csv` | Final submission (64 features, refit on all 14,000 alerts) |
