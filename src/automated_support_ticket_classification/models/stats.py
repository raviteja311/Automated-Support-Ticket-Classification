"""Is a difference in score real, or noise from one particular test set?

A single macro F1 on one test set is a point estimate. These two functions put
error bars on it (bootstrap) and test whether two models on the same test set
really differ (McNemar). The baselines stage calls both.
"""

import numpy as np
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar


def bootstrap_f1_ci(
    y_true,
    y_pred,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for macro F1.

    Resample test-set rows with replacement `n_resamples` times, score macro
    F1 on each resample, and return the (alpha/2, 1 - alpha/2) percentiles,
    rounded to 4 places. Same seed, same interval.

    Rows are resampled as (true, predicted) pairs, never the two columns
    separately: the unit of evidence is one message and what the model said
    about it. The model is not refitted, so the interval covers test-set
    sampling noise only, not training variance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    # One generator for the whole loop: reseeding inside it would draw the
    # same resample every time.
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = rng.integers(0, n, size=n)
        # zero_division=0: a resample that predicts a class it does not
        # contain scores 0 for that class, which is the honest answer, and
        # does not flood the log with warnings.
        scores[i] = f1_score(y_true[rows], y_pred[rows], average="macro", zero_division=0)
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(low), 4), round(float(high), 4)


def mcnemar_p(y_true, pred_a, pred_b) -> float:
    """McNemar test p-value: do models A and B make different errors?

    Only rows where exactly one model is right carry information. Build the
    2x2 table of (A right?, B right?) and pass it to
    statsmodels.stats.contingency_tables.mcnemar. Return the p-value as a
    plain float, unrounded: a tiny p rounded to 4 places becomes 0.0, which
    reads as "impossible" rather than "very unlikely".

    exact=True: the p-value comes from the binomial distribution of the
    discordant rows (under no difference, each is equally likely to favour A
    or B). It is valid at any count, while the chi-square version is an
    approximation that needs roughly 25 or more discordant rows, and its
    continuity correction is undefined when there are none at all.
    """
    y_true = np.asarray(y_true)
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true
    table = [
        [int(np.sum(a_right & b_right)), int(np.sum(a_right & ~b_right))],
        [int(np.sum(~a_right & b_right)), int(np.sum(~a_right & ~b_right))],
    ]
    return float(mcnemar(table, exact=True).pvalue)
