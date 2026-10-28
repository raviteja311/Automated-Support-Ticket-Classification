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

From `python -m automated_support_ticket_classification.models.compare`, run on
2026-10-28 against the rule above (rule SHA-256 `35ec183f...`, recorded in
[metrics/model_comparison.json](../metrics/model_comparison.json)). Reliability
diagram: [reports/reliability.png](../reports/reliability.png).

| model | macro F1 (95% CI) | ECE | p95 latency |
|---|---|---|---|
| logreg | 0.9086 (0.8971 to 0.9195) | 0.155 | 0.409 ms |
| linearsvc_calibrated | 0.9313 (0.9214 to 0.9399) | 0.0359 | 2.39 ms |

McNemar p: 7.25e-10

Each condition of the rule:

1. Macro F1: 0.9313 - 0.9086 = 0.0227, at least 0.01. Passes.
2. McNemar p: 7.25e-10, below 0.05. Passes.
3. ECE: 0.0359, no more than 0.155 + 0.02 = 0.175. Passes.
4. Latency: p95 2.39 ms, at most 5 ms. Passes.

LogReg's high ECE comes from underconfidence: in every bin of the reliability
diagram its accuracy is well above its stated confidence. The calibrated SVC
stays much closer to the diagonal.

## Decision

`linearsvc_calibrated` (sigmoid) ships: it passes all four conditions.

Not applied yet. The switch is a separate change: set
`model.type: linearsvc_calibrated` in `params.yaml`, run `dvc repro`, promote
the new version through the gate in `models/promote.py`, and update the
figures in the README and docs. If the gate refuses, record why here.
