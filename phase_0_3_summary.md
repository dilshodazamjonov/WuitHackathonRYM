# Phases 0–3 summary: data, validation, baseline and behavioral EDA

**Team RYM** · WIUT Hackathon 2026, FinTech track · status as of Fri 25 Sep 2026

Phases 0 (setup), 1 (data audit), 2 (target, split and validation) and 3 (behavioral EDA) are done. Phase 4 (EDA-driven features) starts next. Every number below comes from `notebooks/solution.ipynb`, which runs top to bottom in about a minute and gives identical outputs on every run. The two numbers marked *side check* come from quick checks outside the notebook.

## In short

- **The data is clean.** It has no nulls, no duplicate rows, no orphan transactions and no alerts without a history, so no cleaning step is needed.
- **The test set is a random 30% of alerts from the same two years.** It is not a later period. Test and train cannot be told apart (adversarial AUC 0.497), so ordinary stratified cross-validation estimates the hidden score.
- **Escalation is stable.** 17.2% of alerts are escalated, and the rate does not drift over 24 months. Workload, weekday, month-end and public holidays have no effect.
- **Each history has the same shape.** Transactions are observed within a common 180-day lookback window: slowly declining background activity, then a dense burst in the last minutes before the alert.
- **The burst is structure, not signal.** Phase 3 tested it as the main behavioral hypothesis. Neither the burst itself nor its change from the alert's background separates escalated from dismissed alerts on its own. The review's 13 trigger variables score AUC 0.470–0.528, and all 13 together reach only 0.547.
- **The signal is in transfer amounts.** Escalated alerts make smaller bank transfers, at every point in the six months, while card and cash amounts barely differ. Measured against the alert's own card spending, the gap is the strongest single signal found (AUC 0.397, i.e. 0.603 in the other direction): escalation falls from about 24% to 9% across its deciles.
- **Baseline: AUC 0.614 (Gini 0.228) on the development folds.** It uses only totals over the 180-day window. Very few trees and a best single total of 0.556 show we are feature-limited, not model-limited. A new feature family is kept when adding it improves this OOF AUC on the frozen folds.
- **A valid submission file already exists:** `submissions/team_486052EC.csv`. It passes every format check.

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
- **Expected lift** (*side check*, not in the notebook): LightGBM on the per-channel background amounts alone reached a CV AUC of 0.630, above the 0.614 baseline. This is an expectation for Phase 4, not a result.

### Decisions logged

- **Trigger boundary:** `lag_days <= 1` for counts, shares and amounts. Its AUCs are within 0.002 of the last-hour core, and it also catches the 10 early bursts. The last-hour core is used for timing.
- **Background windows:** no window is special, so 7, 30 and 90 days stay as default candidates only.
- **Error analysis** of the model's most confident misses waits for the first behavioral model.

## What comes next

| When | Phase | Main question |
|---|---|---|
| Sat | 4: EDA-driven features | One feature builder for train and test, in priority order (below). Keep a family when adding it improves the baseline model's OOF AUC on the frozen folds (mean up and at least 4 of 5 folds up). A single feature does not have to beat 0.614 on its own |
| Sat evening | 5: Modeling | LightGBM + CatBoost, light tuning, rank blend; error analysis on the dev out-of-fold predictions; freeze at 22:00 |
| Sun 09:00 | 6: Holdout check, once | Development vs holdout AUC against the warning rule |
| Sun | 6: Final fit and submission | Retrain on all 14,000, validate the CSV, reproduce on a second laptop, submit by 18:00 (deadline 23:59) |

**Phase 4 priorities, from the Phase 3 evidence**

1. **Channel-relative amounts** (main family): per-channel background amount levels and upper-tail shares, plus cross-channel relative sizes, above all transfers against the alert's own card level.
2. **Activity level:** background counts per channel.
3. **Typology:** a small family for in-to-out timing and outgoing shares.
4. **Low priority, tested but not expected to add much:** a compact trigger family, burst compression, channel-mix changes and sequences.

The key question for Phase 4 is whether the channel-relative amount family lifts the OOF AUC clearly above 0.614 on the frozen folds.

## For the team

**Website (EDA site)**
- Figures ready in `figures/`, 14 in total:
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
- Each figure's finding and action are in `artifacts/insights.json`.
- Suggested placement:
  - F04 and F09 go together in the Target and behavior section: the hypothesis we tested and what the data showed instead.
  - F03, F06, F07, F08, F10 and F11 fit the same section as supporting views.
  - F05 fits the transaction history section.
- Publish only aggregated figures and numbers: never raw data, never `split.csv`.

**QA and submission**
- `submissions/team_486052EC.csv` is a valid fallback made from the baseline. The final model will overwrite it.
- The notebook's validator checks columns, row count, IDs, missing values, the 0–1 range and the file name.
- **Open item:** confirm with the organisers that `486052EC` is our TEAM_ID before we submit.

**Files produced so far**

| File | What it is |
|---|---|
| `notebooks/solution.ipynb` | The notebook: Setup, Load, Data audit, Target / split / validation, Behavioural EDA, Exports |
| `artifacts/F00_dataset_overview.csv`, `F00_schema.csv` | Dataset overview and column descriptions |
| `artifacts/split.csv` | Frozen split and folds (internal only) |
| `artifacts/cv_log.csv` | Every model run with its fold scores |
| `artifacts/insights.json` | Figure findings for the website |
| `artifacts/decisions.md` | Every decision with its evidence |
| `figures/F00a`–`F11` | Fourteen figures (PNG) |
| `submissions/team_486052EC.csv` | Baseline safety submission |
