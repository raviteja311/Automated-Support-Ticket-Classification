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

---

## E3 - Real support text, and bigrams finally earning their keep

**Motivation.** E1 compared unigrams against bigrams on synthetic template data
and found no difference at all. That is a suspicious result: it says the feature
space does not matter, which is really a statement about the data rather than
the model. Template text has no phrase-level ambiguity for bigrams to resolve.
The honest way to test the question is on text a human wrote.

**Choosing a corpus.** The obvious candidate, the CFPB consumer-complaints
database, turned out to be unusable. Its 347 MB bulk export contains no
complaint narrative, and neither does its search API, which rejects
`complaint_what_happened` as an invalid field. Both were checked directly. The
dataset has categories but no text, so there is nothing to classify.

Replaced with **banking77**: 13,083 real customer-support messages labelled with
77 fine-grained intents, 1 MB. Its intents fold onto this project's five
categories, including genuine card-delivery intents, which is what makes a
`shipping` class possible at all. The mapping lives in `data/banking77.py` with
borderline calls marked; it is a judgement and a reviewer may disagree with any
of them.

**Resulting class balance.** Deliberately not resampled:

| class | share |
|---|---|
| billing | 37.8% |
| account | 22.6% |
| technical | 16.6% |
| general | 15.4% |
| shipping | 7.6% |

Real support traffic is skewed. Flattening it would hide exactly the effect
macro F1 exists to measure.

**Method.** `data.source` is now a DVC-tracked param, so switching corpus reruns
the pipeline. Three forced runs on identical data, one variable changed:

| ngram_max | accuracy | f1_macro | decision |
|-----------|----------|----------|----------|
| 1         | 0.9201   | 0.9100   | baseline |
| 2         | **0.9282** | **0.9171** | **winner, adopted** |
| 3         | 0.9251   | 0.9139   | worse than 2 |

**Result.** Bigrams win: +0.81pp accuracy, +0.71pp macro F1. Trigrams give the
gain back, which is the usual shape of an n-gram sweep: more context helps until
the features get too sparse to estimate.

**Decision.** `ngram_max: 2`. This reverses E1, and the reversal is the point.
The same experiment on different data gives a different answer, so E1's
conclusion was never about bigrams, it was about the dataset. An experiment on
data that cannot distinguish two options will always report that the two options
are the same.

**Per-class, on real text:**

| class | precision | recall | f1 | support |
|-----------|-----------|--------|-------|---------|
| billing   | 0.941 | 0.970 | 0.955 | 989 |
| technical | 0.919 | 0.915 | 0.917 | 435 |
| account   | 0.932 | 0.927 | 0.930 | 592 |
| shipping  | 0.930 | 0.874 | 0.901 | 198 |
| general   | 0.897 | 0.868 | 0.883 | 403 |

`general` is now the weakest class, which makes sense: it is the residual
category, holding whatever did not fit elsewhere, so it has the least coherent
vocabulary. On synthetic data it scored 0.922 because the templates gave it a
tidy word list no real catch-all category has.

Note the scores are *lower* than the synthetic 0.9300 / 0.9297. That is expected
and is the whole point. The synthetic number measured how well a model can learn
five template sets, which is not a hard question.

---

## E4 - The promotion gate was comparing incomparable numbers

**Not a planned experiment.** Found by running the E3 model through the
promotion gate added alongside the model registry.

**Symptom.** The real-data model scored macro F1 0.9171. The gate refused to
promote it, because production held a synthetic-data model scoring 0.9297.

**Why that is wrong.** Those two numbers come from different test sets. One
measures performance on 2,617 real support messages, the other on 600 generated
from templates. Comparing them is meaningless, and the gate was confidently
making a shipping decision on a meaningless comparison. It happened to refuse,
which looks like caution but was luck: had the synthetic number been lower, it
would have shipped the worse model just as confidently.

**Fix.** Every registered version is now tagged with the corpus that produced
its score, and the gate only compares like with like. When the corpus changes,
the incumbent is not evidence of anything, so the candidate takes over on the
new corpus and future comparisons proceed normally from there.

**Lesson.** A metric gate is only as meaningful as the comparability of the two
numbers it compares. "Is the new model better?" is not answerable unless both
models were measured the same way, and nothing in a plain float carries that
context. Tag the score with what produced it, or the gate is theatre.

---

## E5 - The drift monitor was crying wolf on tiny samples

**Not a planned experiment.** Spotted while reading the first real drift report
after the monitor went in.

**Symptom.** With five predictions in the log, the monitor reported label drift
of 0.1764 against a 0.10 threshold, and flagged it. Later, with seventeen rows,
0.2662. Both looked like a model whose output distribution had shifted.

**Why it was wrong.** Neither was a signal. Drift tests compare distributions,
and a distribution estimated from a handful of requests is mostly noise. Five
predictions spread one per category look dramatically "drifted" against a
training set that is 38% billing and 8% shipping, purely because five is far too
few to estimate a share from. The monitor was not detecting a change in the
world; it was detecting that it had barely looked at it.

Left alone this is worse than no monitor. It would fire on the first few
requests after every deploy, and the first thing anyone does with an alert that
is usually wrong is stop reading it.

**Fix.** A minimum-sample gate, `MIN_SAMPLE_ROWS = 200`, chosen so the natural
class balance still leaves roughly 40 rows in the smallest class. Below it, the
scores are still computed and reported, because they are the evidence, but no
verdict is issued at all.

The distinction that matters is between *no drift* and *do not know*:

| rows | status | dataset_drifted |
|---|---|---|
| 17 | `insufficient_data` | `null` |
| 300 | `ok` | `false` |

`dataset_drifted` is null rather than false on a thin sample, and a test pins
that difference. A monitor keying on `dataset_drifted` alone would read an
unknown as a clean bill of health, which is how a monitor goes silent without
anyone noticing. The rule is: alert only when `status == "ok"` and
`dataset_drifted` is true.

**Verified both ways on real traffic.** 17 logged predictions with a label score
of 0.2662, well past the threshold, now yields no verdict. 300 rows of realistic
traffic yields `status: ok` with both columns correctly reading no drift.

**Lesson.** Every statistical alarm needs a floor on the evidence it is allowed
to fire from. A threshold answers "how big a difference matters"; it says
nothing about "how much data before the question is worth asking". Both are
needed, and only one of them tends to get written down.
