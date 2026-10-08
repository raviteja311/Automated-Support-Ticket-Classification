# Model decision: LogReg or calibrated LinearSVC

Write the rule **before** running `models/compare.py`, commit it, and only then
run the comparison. The script refuses to run while this file contains the word
TODO, and records this file's SHA-256 in `metrics/model_comparison.json`, so
anyone can check the rule was not edited after the numbers came in.

## Rule

TODO: write the rule in your own words. It must say, in advance:

- which numbers decide (macro F1, McNemar p, ECE, latency) and the threshold
  for each
- what happens on a tie or a mixed result
- which calibration method you will use (`model.calibration`: sigmoid or
  isotonic) and why, decided now rather than by trying both

An example of the shape, not a recommendation: "Switch if calibrated SVC's
macro F1 is higher with McNemar p < 0.05 and its ECE is no worse than
LogReg's + 0.02. Otherwise keep LogReg."

Written on: TODO (date)

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
