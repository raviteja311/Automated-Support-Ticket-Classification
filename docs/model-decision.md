# Model decision: LogReg or calibrated LinearSVC

Write the rule **before** running `models/compare.py`, commit it, and only then
run the comparison. The script refuses to run while this file still has its
placeholder markers, and records this file's SHA-256 in `metrics/model_comparison.json`, so
anyone can check the rule was not edited after the numbers came in.

## Rule

Switch production to `linearsvc_calibrated` only if all four conditions hold
on the official test split (3,080 rows). Otherwise keep `logreg`.

1. **Macro F1:** calibrated SVC's macro F1 is at least 0.01 higher than
   LogReg's. This is the same 1pp "clear win" margin the baselines stage uses.
2. **McNemar p:** below 0.05, so the difference in errors is not chance.
3. **ECE:** calibrated SVC's ECE is no more than LogReg's ECE + 0.02, so its
   confidences are about as trustworthy for the review threshold.
4. **Latency:** calibrated SVC's p95 single-message `predict_proba` latency is
   at most 5 ms. The whole API call takes about 2.6 ms, so this only rules out
   a large slowdown.

**Ties and mixed results:** if any condition fails, including an exact tie on
F1, LogReg stays. Switching has a cost (retraining, docs, re-choosing the
review threshold), so the challenger must clear every condition.

**Calibration method:** `sigmoid`, the current `params.yaml` value. With 5-fold
calibration there are about 2,000 rows per fold, and only a couple of hundred
for `card_delivery`. That's too few to trust isotonic's free-form fit, while
sigmoid fits two parameters per class and stays stable.

Written on: 2026-10-28

## Result

Fill in after running `python -m automated_support_ticket_classification.models.compare`.

| model | macro F1 (95% CI) | ECE | p95 latency |
|---|---|---|---|
| logreg | | | |
| linearsvc_calibrated | | | |

McNemar p:

## Decision

Which model ships, and why, by the rule above. If it is the calibrated SVC, set
`model.type: linearsvc_calibrated` in `params.yaml`, run `dvc repro`, and
promote the new version through the gate in `models/promote.py`. If the gate
refuses, record why here.
