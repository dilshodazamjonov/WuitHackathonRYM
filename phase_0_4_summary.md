# Phases 0–4 summary: data, validation, baseline, behavioral EDA and features

**Team RYM** · WIUT Hackathon 2026, FinTech track · status as of Sat 26 Sep 2026

Phases 0 (setup), 1 (data audit), 2 (target, split and validation), 3 (behavioral EDA) and 4 (EDA-driven features) are done. Phase 5 (models and blend) starts next. Every number below comes from `notebooks/solution.ipynb`, which runs top to bottom in about three minutes and gives identical outputs on every run. Numbers marked *side check* come from quick checks outside the notebook.

## In short

- **The data is clean.** It has no nulls, no duplicate rows, no orphan transactions and no alerts without a history, so no cleaning step is needed.
- **The test set is a random 30% of alerts from the same two years.** It is not a later period. Test and train cannot be told apart (adversarial AUC 0.497 on the totals, 0.500 on all 199 features), so ordinary stratified cross-validation estimates the hidden score.
- **Escalation is stable.** 17.2% of alerts are escalated, and the rate does not drift over 24 months. Workload, weekday, month-end and public holidays have no effect.
- **Each history has the same shape.** Transactions are observed within a common 180-day lookback window: slowly declining background activity, then a dense burst in the last minutes before the alert.
- **The burst is structure, not signal.** Phase 3 tested it as the main behavioral hypothesis: neither the burst nor its change from the alert's background separates the classes on its own (13 trigger variables at AUC 0.470–0.528). Phase 4 confirmed it inside the model: the trigger family adds −0.0001 AUC on top of the amounts.
- **The signal is in amounts measured on their own channel's scale.** Escalated alerts make bank transfers that are small for their own card spending (AUC 0.397). Phase 4 found the mirror image in cash: incoming cash that is large for the alert's card level goes with escalation (AUC 0.549).
- **Features lift the model from 0.614 to 0.637** (Gini 0.228 → 0.274) on the development folds. Almost all of the gain comes from one family, the channel-relative amounts (+0.019). Six of the nine families tested add nothing and are dropped. The frozen set has 72 features.
- **The model's confident errors are mirror images of the other class.** They are not a pattern a missing feature would catch.
- **A valid submission file already exists:** `submissions/team_486052EC.csv`, from the baseline. It passes every format check.

## The data

| | Train | Test |
|---|---|---|
| Alerts | 14,000 (with labels) | 6,000 |
| Transactions | 6,987,663 | 3,027,575 |
| Alert dates | 1 Jan 2025 – 31 Dec 2026 | same range |
| Transactions per alert | median 461, max 2,279 | median 460.5, max 1,985 |
| Lookback window | a common 180 days before the alert date | same |

Each alert has one row in the signals table: its ID, alert date and, for train, the label `eskalatsiya` (1 = escalated, 0 = dismissed). Each transaction has a timestamp, a direction (incoming or outgoing), a type (card, bank transfer, cash, international) and `miqdor_indeksi`, an amount index on an unknown scale.

The submission has one row per test alert with the escalation probability. It is scored by ROC-AUC.

## What the audit found (Phase 1)

| Question | Finding | What we do |
|---|---|---|
| Is the data clean? | 0 nulls, 0 duplicate rows, 0 orphan transactions, 0 alerts without a history | No cleaning step |
| Do train and test look alike? | Category shares agree within 0.2 points and amount quantiles agree to two decimals | Same pipeline for both |
| Do IDs or file order leak the label? | ID number, file row order and alert date all score AUC 0.498–0.508; the test share is 28–32% in every ID decile | Never used as features |
| Do alerts share customers? | 0 of 10,015,238 transaction rows appear under more than one alert | No group-aware folds, no linkage features |
| Is anything recorded after the alert? | 0 transactions after the alert date; transactions are observed within a common 180-day lookback window (the earliest transaction sits 180 days out for the median alert, at least 170 days out for 95%) | No leakage filter needed |
| Is there a daily or weekly rhythm? | Hours 0–22 hold 3.8% each and weekdays 14.3% each. Hour 23 holds 11.7%, and that is the pre-alert burst | No hour or weekday features |
| Where is the activity? | Background falls from about 3 transactions a day (120 days out) to about 1 a day in the last week. Then about 8% of all rows arrive in a burst: median 31 per alert, 1.5 minutes before the alert midnight, 99% within 3 minutes | Treat the trigger window (`lag_days <= 1`) and the background (`lag_days >= 2`) separately |
| Do all alerts have a burst? | 152 alerts have none; 21 have it just after midnight on the alert day itself | The window is defined by day (`lag_days <= 1`), not by clock time |
| What is the amount index? | One global scale, near-normal (mean −0.13, sd 0.98), not standardized per alert (per-alert means spread with sd 0.49). Each channel is its own bell curve: card lowest, international about 2 sd higher, outgoing above incoming | Amounts are compared within their own channel (direction × type), not on one global threshold. `exp(index)` is only an experiment |
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
- **What a warning means:** we check the development folds and prefer the simpler model, and we never tune on the holdout. The rule is a warning threshold, not proof that the model generalizes.
- **Final model:** the same frozen configuration, retrained with 5 folds on all 14,000 alerts. Test predictions are averaged over the folds.

### Is escalation stable, and is the test set like train?

| Check (development set only, where labels are used) | Result | Conclusion |
|---|---|---|
| Monthly escalation rate, 24 months | 13.6%–20.7%, no step or trend; chi-square p = 0.14; 2025 at 17.0% vs 2026 at 17.3% | Policy is stable, so no time-ordered validation |
| Alert volume vs rate | Volume moves in steps (up about 60% in July 2025, down in April and July 2026); alerts per day vs escalation gives AUC 0.49 | Busy days don't change outcomes; not a feature |
| Calendar | Weekday p = 0.45; last 3 days of the month p = 0.63; ±3 days around Navruz, Independence Day and both hayits p = 0.53 | No date or calendar features |
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
| Test predictions (safety file) | 6,000 rows, 0.118 to 0.324, median 0.165 |

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
- **The trigger window and the burst differ in length.** The burst spans a median 2.8 minutes, but the trigger window spans 135 minutes, because 7,200 alerts also have a few ordinary transactions earlier on the day before.
- **Edge cases:**
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
- **It sharpens against the alert's own level.** An alert's amounts move together across channels (Spearman 0.47–0.87 between its channel means). Outgoing transfer size minus outgoing card size scores AUC 0.397, with every fold between 0.375 and 0.430, while card size alone scores 0.495.
- **The interaction is visible directly.** In the middle card-size quintile, escalation falls from 25% for the smallest transfers to 4% for the largest. For mid-sized transfers, it rises from 13% with the smallest card payments to 24% with the largest.
- **Across deciles** of the relative measure, the escalation rate runs from about 24% down to 9.3%, about 2.5 times apart.
- **How we read it:** escalated alerts move money by bank transfer in smaller amounts than their own card spending would suggest. The pattern recalls structuring, but the data cannot say why analysts escalate it.
- **Confirmed in Phase 4:** adding the amounts family to the baseline lifts the development CV AUC from 0.614 to 0.633 (see below).

### Decisions logged

- **Trigger boundary:** `lag_days <= 1` for counts, shares and amounts. Its AUCs are within 0.002 of the last-hour core, and it also catches the 10 early bursts. The last-hour core is used for timing.
- **Background windows:** no window is special, so 7, 30 and 90 days stay as default candidates only.
- **Error analysis** of the model's most confident misses: done in Phase 4, on the first behavioral model.

## Features and ablation (Phase 4)

### How the features were built

- **One builder for train and test.** `build_features()` turns every history into one row with the same code for both sets.
- **No labels in the features.** Channel statistics (mean, sd, p95, p99 and the per-type cap for each direction × type) come from train and test transactions together.
- **The baseline stays in.** The 29 totals remain as the reference, and every other feature belongs to one family, so the ablation keeps or drops each family as a whole.
- **199 features in total.** None is constant or a copy of another, and missing values stay missing rather than imputed.
- **Left out on purpose:** alerts per day and the alert-date calendar, because Phase 2 showed they carry no signal.

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

**Frozen feature set (`FEATURES`):** the 29 totals plus amounts, mix and compression, 72 columns.

| Metric | Baseline (29 totals) | Kept set (72 features) |
|---|---|---|
| **Development CV AUC** | 0.614 ± 0.016 | **0.637 ± 0.028** |
| **Gini** | 0.228 | **0.274** |
| Fold AUCs | 0.617 · 0.625 · 0.611 · 0.632 · 0.585 | 0.640 · 0.665 · 0.609 · 0.670 · 0.601 |

### What the kept model uses

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
5. **Two families were kept by a hair.** Mix and compression each add +0.002. A side check with other seeds (below) shows the compression gain does not hold up.
6. **International transfers are strong but mostly missing.** Outgoing international minus card size scores 0.434, one of the strongest features. But two thirds of alerts (67%) never use the channel, so it can help only a third of the queue.
7. **The fold spread grew.** The kept set's fold AUCs range from 0.601 to 0.670 (± 0.028), against ± 0.016 for the baseline. The amounts gain varies by fold, from −0.002 on the third fold to +0.034 on the second. The fifth fold stays the weakest, as it was for the baseline.
8. **No drift on any of the 199 features.** This supports the Phase 2 finding that the test set is the same population.

### Error analysis: the confident misses are mirror images

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

**Seed stability of the kept families.** The same ablation steps, rerun with LightGBM seeds 7 and 2026 on the same folds:

| Seed | Totals | + amounts | + mix | + compression |
|---|---|---|---|---|
| 42 (notebook) | 0.6139 | 0.6326 | 0.6348 | 0.6369 |
| 7 | 0.6113 | 0.6321 | 0.6350 | 0.6341 |
| 2026 | 0.6110 | 0.6311 | 0.6356 | 0.6310 |

- **Amounts holds up.** Its gain is +0.019 to +0.021 under every seed.
- **Mix holds up too, though it is small:** +0.002 to +0.005.
- **Compression does not.** It gives +0.002, −0.001 and −0.005, so its place in the kept set is one-seed luck.
- **Amounts alone beats the totals.** The 30 amounts features without the totals reach 0.632.
- **Shrinkage does not help.** Shrinking the per-alert channel means towards the channel average for short histories (the missed escalations are short) scores 0.635 against 0.637: no effect.

### Decisions

- **`FEATURES` frozen by the rule:** the 29 totals plus amounts, mix and compression, 72 columns, at a development CV AUC of 0.637.
- **Dropped:** windows, typology, trigger, rhythm, sequence and context. Each lowered the mean fold AUC.
- **Open for the team:** compression passes the pre-registered rule but fails the seed side check. Either keep it (the rule as written), or recheck mix and compression with the three final seeds in Phase 5 and drop what does not hold.

## What comes next

| When | Phase | Main question |
|---|---|---|
| Sat evening | 5: Modeling | LightGBM + CatBoost on the 72 features, light tuning, rank blend, stability report; settle the compression question; freeze at 22:00 |
| Sun 09:00 | 6: Holdout check, once | Development vs holdout AUC against the warning rule |
| Sun | 6: Final fit and submission | Retrain on all 14,000, validate the CSV, reproduce on a second laptop, submit by 18:00 (deadline 23:59) |

The baseline was feature-limited (5–86 trees per fold). With the new features, Phase 5 checks whether the model side can add anything on top: CatBoost as a blend partner, light tuning, and seed averaging.

## For the team

**Website (EDA site)**
- Figures ready in `figures/`, 18 in total:
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
  - Features (new):
    - F12 the six features with the most gain, escalation rate by decile
    - F13 family ablation: which EDA ideas paid off (**key figure**)
    - F14 the 25 strongest single features
    - F15 drift: PSI and the train-vs-test classifier on all features
- Each figure's finding and action are in `artifacts/insights.json`.
- New tables for the website: `artifacts/feature_screen.csv` (every feature's AUC, fold range, IV, family) and `artifacts/ablation.csv`.
- Suggested placement:
  - Target and behavior section: F04 and F09 go together (the hypothesis we tested and what the data showed instead). F03, F06, F07, F08, F10 and F11 fit the same section as supporting views.
  - Transaction history section: F05.
  - "Features and modeling ideas motivated by EDA" section: F13, then F12 and F14. F13 is the story in one chart: the amounts pay off, and the burst, windows and sequences do not.
  - Validation section: F15, next to F02.
- Publish only aggregated figures and numbers: never raw data, never `split.csv`.

**QA and submission**
- `submissions/team_486052EC.csv` is still the valid fallback made from the baseline. The final model will overwrite it in Phase 6.
- The notebook's validator checks columns, row count, IDs, missing values, the 0–1 range and the file name.
- A full run takes about three minutes; two consecutive runs give identical md5 for every figure, artifact and the submission.
- **Open item:** confirm with the organisers that `486052EC` is our TEAM_ID before we submit.

**Files produced so far**

| File | What it is |
|---|---|
| `notebooks/solution.ipynb` | The notebook: Setup, Load, Data audit, Target / split / validation, Behavioural EDA, Feature engineering, Screening / drift / ablation, Exports |
| `artifacts/F00_dataset_overview.csv`, `F00_schema.csv` | Dataset overview and column descriptions |
| `artifacts/split.csv` | Frozen split and folds (internal only) |
| `artifacts/cv_log.csv` | Every model run with its fold scores, including each ablation step |
| `artifacts/feature_screen.csv` | Univariate screen of all 199 features |
| `artifacts/ablation.csv` | Family ablation table |
| `artifacts/insights.json` | Figure findings for the website |
| `artifacts/decisions.md` | Every decision with its evidence |
| `figures/F00a`–`F15` | Eighteen figures (PNG) |
| `submissions/team_486052EC.csv` | Baseline safety submission |
