"""Is a difference in score real, or noise from one particular test set?

A single macro F1 on one test set is a point estimate. These two functions put
error bars on it (bootstrap) and test whether two models on the same test set
really differ (McNemar). The baselines stage calls both.

YOUR CODE: both functions are stubs. tests/test_stats.py says what they must
do; remove the xfail marker there once each one is implemented.
"""

import numpy as np


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

    Hints: np.random.default_rng(seed), rng.integers, np.percentile.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    raise NotImplementedError("bootstrap_f1_ci: write this in models/stats.py")


def mcnemar_p(y_true, pred_a, pred_b) -> float:
    """McNemar test p-value: do models A and B make different errors?

    Only rows where exactly one model is right carry information. Build the
    2x2 table of (A right?, B right?) and pass it to
    statsmodels.stats.contingency_tables.mcnemar. Return the p-value as a
    plain float, unrounded: a tiny p rounded to 4 places becomes 0.0, which
    reads as "impossible" rather than "very unlikely".

    Think about which variant to use (exact=True or the chi-square version
    with a continuity correction) and be ready to say why.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    raise NotImplementedError("mcnemar_p: write this in models/stats.py")
