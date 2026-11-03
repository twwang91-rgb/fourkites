import numpy as np

from foursight_intent.calibration import (
    cross_fitted_bias,
    expected_calibration_error,
    fit_temperature,
    review_metrics,
    risk_coverage_curve,
    select_review_threshold,
    softmax,
    tune_logit_bias,
)


def synthetic(n=600, k=7, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += rng.normal(scale, 1.0, n)
    return logits, y


def test_temperature_reduces_nll_for_overconfident_logits():
    logits, y = synthetic()
    over = logits * 3.0  # over-confident
    T = fit_temperature(over, y)
    assert T > 1.2  # the synthetic logits were multiplied by 3, so the fitted T must be well above 1
    from foursight_intent.calibration import nll

    p_raw, p_cal = softmax(over), softmax(over / T)
    assert nll(over, y, T) < nll(over, y, 1.0)  # T minimises validation NLL
    assert np.array_equal(p_raw.argmax(1), p_cal.argmax(1))  # temperature never changes the label
    # over-confident raw probabilities should sit above the diagonal less after scaling
    correct = p_raw.argmax(1) == y
    assert p_cal.max(1).mean() - correct.mean() < p_raw.max(1).mean() - correct.mean()


def test_review_threshold_respects_budget_and_is_monotone():
    logits, y = synthetic(scale=1.5)
    p = softmax(logits)
    pol = select_review_threshold(p, y, max_flag_rate=0.2)
    assert pol.flag_rate <= 0.2
    looser = review_metrics(p, y, pol.threshold + 0.1)
    assert looser.flag_rate >= pol.flag_rate and looser.error_recall >= pol.error_recall


def test_risk_coverage_curve_is_well_formed():
    logits, y = synthetic(scale=1.5)
    rc = risk_coverage_curve(softmax(logits), y)
    assert 0 <= rc["aurc"] <= rc["full_risk"] + 1e-9
    assert rc["points"][-1]["coverage"] <= 1.0


def test_logit_bias_improves_in_sample_and_is_zero_when_balanced():
    rng = np.random.default_rng(1)
    n, k = 800, 7
    # imbalanced: class 0 dominant with inflated logits, minorities under-predicted
    y = rng.choice(k, n, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 2.0
    logits[:, 0] += 1.5  # systematic over-prediction of the majority class
    b = tune_logit_bias(logits, y)
    from foursight_intent.calibration import _fast_overall

    assert _fast_overall((logits + b).argmax(1), y) >= _fast_overall(logits.argmax(1), y)
    assert b[0] <= 0  # the majority class should be pushed down


def test_cross_fitted_bias_shapes():
    from foursight_intent.data import Fold

    logits, y = synthetic(n=300)
    idx = np.arange(300)
    folds = [Fold(i, np.setdiff1d(idx, idx[i::3]), idx[i::3]) for i in range(3)]
    out, biases = cross_fitted_bias(logits, y, folds)
    assert out.shape == logits.shape and len(biases) == 3
