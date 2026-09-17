# Experiments log

## E1 - Do bigrams beat unigrams?

**Hypothesis.** Adding bigrams to the TF-IDF vectoriser gives the model access to short
phrases such as "not working" or "double charge", which should separate the `technical`
and `billing` classes more cleanly than single tokens alone.

**Method.** One variable changed: `model.ngram_max` in `params.yaml`, 1 then 2. Everything
else held fixed, including `data.random_state: 42`, so the train/test split is identical
across both runs. Both runs tracked in MLflow under the `ticket-triage` experiment.

| Run | ngram_max | accuracy | f1_macro | decision |
|-----|-----------|----------|----------|----------|
| 1   | 1         | 0.9217   | 0.9212   | baseline |
| 2   | 2         | 0.9217   | 0.9212   | no change, reverted |

**Result.** Identical to four decimal places. Bigrams bought nothing at all.

**Decision.** Keep `ngram_max: 1`.

**Why this is the right call.** The two configurations score the same, so the tie breaks on
cost. Bigrams inflate the vocabulary substantially for no measurable gain, which means a
larger vectoriser, a larger `model.joblib`, and a larger Docker image, all to make the same
predictions. When scores tie, take the smaller, faster, more interpretable model.

**Why the result is unsurprising in hindsight.** The dataset is generated from a small set
of templates with a fixed filler vocabulary, and the class signal sits in individual content
words: "refund", "invoice", "crash", "password", "tracking". There is no phrase-level
ambiguity for bigrams to resolve. The 10 percent label noise injected by the generator also
puts a ceiling on achievable accuracy that no feature change can lift, which is consistent
with both runs landing in the low 0.92s rather than near 1.0.

**What would actually move the number.** Real ticket text, where phrasing carries meaning
that single tokens lose. That is the point at which bigrams, or a transformer, earn their
cost. On synthetic template data, neither would.
