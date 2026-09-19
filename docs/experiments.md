# Experiments log

## E1 - Do bigrams beat unigrams?

**Hypothesis.** Adding bigrams to the TF-IDF vectoriser gives the model access to short
phrases such as "not working" or "double charge", which should separate the `technical`
and `billing` classes more cleanly than single tokens alone.

**Method.** One variable changed: `model.ngram_max` in `params.yaml`, 1 then 2. Everything
else held fixed, including `data.random_state: 42`, so the train/test split is identical
across both runs. Both runs tracked in MLflow.

| Run | ngram_max | accuracy | f1_macro | decision |
|-----|-----------|----------|----------|----------|
| 1   | 1         | 0.9217   | 0.9212   | baseline |
| 2   | 2         | 0.9217   | 0.9212   | no change, reverted |

**Result.** Identical to four decimal places. Bigrams bought nothing.

**Decision.** Keep `ngram_max: 1`. When scores tie, the tie breaks on cost: bigrams inflate
the vocabulary substantially for no measurable gain, which means a larger vectoriser, a
larger `model.joblib`, and a larger Docker image, all to make the same predictions.

> **Caveat, added after E2.** These numbers were measured on a corrupted dataset that had
> only four classes. They remain a valid comparison, because both runs used the same bad
> data and only `ngram_max` changed, but they are not comparable to any figure below.
> The conclusion still holds; E2 did not change it.

---

## E2 - Data defect: a missing class and a contaminated one

**Not a planned experiment.** This was found while reading `dvc metrics show` output after
wiring up the DVC pipeline on Day 17. The per-class table listed only four classes.

**Symptom.** `metrics.json` reported `account`, `billing`, `general` and `technical`, with
supports summing to exactly 600. `shipping` was absent. Separately, `account` had the worst
recall of any class (0.869) and `technical` the worst precision (0.905), which is the
signature of one class's examples being predicted as another.

**Root cause.** A transcription error in `TEMPLATES` in `data/generate.py`, introduced when
the generator was copied out of the PDF guide. The `"shipping": [` key sits at the very
bottom of page 22 of the guide and its five entries resume on page 23, with an unrelated
file-path caption interleaved between them. Two things went wrong at that page break:

1. `shipping` was dropped entirely, so the dataset had four classes instead of five.
2. The five `technical` templates were filed under `account`, giving `account` ten
   templates, half of which describe technical problems. Five unrelated technical
   templates that appear nowhere in the guide had been written to fill the resulting gap.

So `account` was not merely mislabelled at the margin: half of every `account` row was a
technical support ticket wearing the wrong label. The model was not underperforming. It was
being asked to learn a contradiction.

A third, quieter symptom confirmed the diagnosis: the `order` entry in `FILLERS` was dead
code, because only the missing `shipping` templates ever used `{order}`.

**Fix.** Restored the canonical taxonomy: five classes, five templates each, `shipping`
reinstated, and the technical templates moved back out of `account`. The five non-canonical
technical templates were removed in favour of the guide's originals, so template counts are
symmetric across classes.

**Result.** Re-ran with `dvc repro`, which correctly cascaded all four stages because
`generate.py` is a declared dependency of the `generate` stage.

| | before (4 classes, contaminated) | after (5 classes, canonical) |
|---|---|---|
| accuracy | 0.9217 | **0.9300** |
| f1_macro | 0.9212 | **0.9297** |

Per-class, after the fix:

| class | precision | recall | f1 | support |
|-----------|-----------|--------|-------|---------|
| billing   | 0.930 | 0.906 | 0.918 | 117 |
| technical | 0.916 | 0.932 | 0.924 | 117 |
| account   | 0.929 | 0.920 | 0.924 | 113 |
| shipping  | 0.968 | 0.952 | 0.960 | 126 |
| general   | 0.908 | 0.937 | 0.922 | 127 |

**Two things worth noticing.** First, accuracy went *up* despite the task getting harder,
adding a fifth class, because removing the contradictory labels mattered more than the extra
class cost. Second, `account` recall rose from 0.869 to 0.920 and is now unremarkable among
its peers, which is exactly what you would predict if contamination had been the cause.

`shipping` scores highest at 0.960 F1, which makes sense: its templates all contain an
`{order}` token like `#10231`, a lexical signal no other class produces. The remaining
error is dominated by the 10 percent label noise the generator injects deliberately, which
puts a hard ceiling on achievable accuracy no feature change can lift.

**Validation.** The post-fix figures, accuracy 0.9300 and f1_macro 0.9297, match the
expected output printed in the guide for Day 10 exactly. That is independent confirmation
that the dataset now matches the one the guide intends.

**Lesson.** The bug was invisible from the training logs, which happily reported a healthy
0.92, and invisible from the unit tests, which assert on shape and column names rather than
on the label set. It only became visible when per-class metrics were laid out side by side.
That is the argument for Day 12 existing at all: summary accuracy hides a missing class,
and per-class reporting does not.

**Guard added.** `tests/test_data.py` now pins the label set, so this class of error cannot
return silently:

```python
def test_generate_covers_all_five_categories():
    df = generate(n_samples=500, seed=0)
    assert set(df["label"]) == {"billing", "technical", "account", "shipping", "general"}
```
