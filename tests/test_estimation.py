import numpy as np
import pytest

from causalopt.estimation import (
    bw_select,
    rd_estimate,
    rd_objects,
)

# If you already import other helpers in this file, keep them there.
# These tests assume your _prepare_inputs sorts x and recenters c to 0 internally.


# ============================================================
# Shared fixtures / DGPs
# ============================================================


def _make_sharp_rd_data(n=400, tau=2.0, seed=0, add_covariate=False):
    """
    Simple sharp RD DGP with continuous baseline plus jump tau at 0.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n)
    eps = rng.normal(scale=0.2, size=n)

    # smooth baseline
    y0 = 1.0 + 0.5 * x + 0.25 * x**2
    y = y0 + tau * (x >= 0) + eps

    if add_covariate:
        z = rng.normal(size=n)
        # allow z to affect y so covariate adjustment is meaningful
        y = y + 0.7 * z
        return y, x, z.reshape(-1, 1)

    return y, x, None


# ============================================================
# bw_select
# ============================================================


def test_bw_select_structure_and_basic_properties():
    y, x, Z = _make_sharp_rd_data(n=300, tau=1.5, seed=1, add_covariate=False)

    out = bw_select(y, x, c=0, p=1, covariates=Z)

    # required top-level keys
    assert out["bwselect"] == "mserd"
    assert out["kernel"] == "tri"
    assert out["vce"] == "nn"
    assert out["p"] == 1
    assert out["q"] == 2
    assert out["c"] == 0

    # counts
    assert out["N"]["left"] + out["N"]["right"] == len(x)
    assert out["M"]["left"] >= 1
    assert out["M"]["right"] >= 1

    # bandwidths exist, finite, positive, symmetric
    bws = out["bandwidths"]
    for k in ("h_left", "h_right", "b_left", "b_right"):
        assert np.isfinite(bws[k])
        assert bws[k] > 0

    assert bws["h_left"] == pytest.approx(bws["h_right"])
    assert bws["b_left"] == pytest.approx(bws["b_right"])

    # internals
    assert np.isfinite(out["internals"]["c_bw"])
    assert np.isfinite(out["internals"]["d_bw"])
    assert out["internals"]["c_bw"] > 0
    assert out["internals"]["d_bw"] > 0


def test_bw_select_invariant_to_cutoff_shift_and_x_shift():
    y, x, Z = _make_sharp_rd_data(n=350, tau=2.0, seed=2)

    out0 = bw_select(y, x, c=0, p=1)
    a = 0.37
    out1 = bw_select(y, x + a, c=a, p=1)

    for k in ("h_left", "b_left"):
        assert out1["bandwidths"][k] == pytest.approx(out0["bandwidths"][k], rel=1e-9)


def test_bw_select_with_covariates_runs_and_returns_valid():
    y, x, Z = _make_sharp_rd_data(n=320, tau=1.0, seed=3, add_covariate=True)

    out = bw_select(y, x, c=0, p=1, covariates=Z)

    assert out["M"]["left"] >= 1
    assert out["M"]["right"] >= 1
    assert out["bandwidths"]["h_left"] > 0
    assert out["bandwidths"]["b_left"] > 0


def test_bw_select_subset_drops_observations():
    y, x, Z = _make_sharp_rd_data(n=200, tau=1.0, seed=4, add_covariate=False)

    subset = np.arange(0, 200, 2)  # keep half
    out = bw_select(y, x, c=0, p=1, covariates=Z, subset=subset)

    assert out["N"]["left"] + out["N"]["right"] == len(subset)


# ============================================================
# rd_estimate
# ============================================================


def test_rd_estimate_output_structure_and_finite():
    y, x, Z = _make_sharp_rd_data(n=500, tau=2.0, seed=10, add_covariate=False)

    out = rd_estimate(y, x, c=0, p=1, covariates=Z)

    # key structure
    assert set(out["tau"].keys()) == {"conventional", "bias_corrected"}
    assert set(out["se"].keys()) == {"conventional", "robust"}
    assert set(out["ci"].keys()) == {"conventional", "bias_corrected", "robust"}
    assert set(out["bandwidths"].keys()) == {"h_l", "h_r", "b_l", "b_r"}

    # finiteness + ordering
    for k in out["tau"]:
        assert np.isfinite(out["tau"][k])
    for k in out["se"]:
        assert np.isfinite(out["se"][k])
        assert out["se"][k] >= 0

    for k, (lo, hi) in out["ci"].items():
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi

    # bandwidths positive
    for k in out["bandwidths"]:
        assert out["bandwidths"][k] > 0


def test_rd_estimate_recovers_known_jump_reasonably():
    # Not a strict equality test; just checks the estimator is in the ballpark.
    tau_true = 1.75
    y, x, Z = _make_sharp_rd_data(n=1200, tau=tau_true, seed=11, add_covariate=False)

    out = rd_estimate(y, x, c=0, p=1, covariates=Z)

    tau_hat = out["tau"]["bias_corrected"]
    # This tolerance is intentionally loose to avoid brittleness across environments.
    assert tau_hat == pytest.approx(tau_true, abs=0.25)


def test_rd_estimate_with_covariates_runs_and_reasonable():
    tau_true = 2.0
    y, x, Z = _make_sharp_rd_data(n=1200, tau=tau_true, seed=12, add_covariate=True)

    out = rd_estimate(y, x, c=0, p=1, covariates=Z)

    # still in the ballpark
    assert out["tau"]["bias_corrected"] == pytest.approx(tau_true, abs=0.35)

    # robust CI should contain bias-corrected point estimate (by construction)
    lo, hi = out["ci"]["robust"]
    assert lo <= out["tau"]["bias_corrected"] <= hi


def test_rd_estimate_cutoff_shift_invariant():
    tau_true = 1.0
    y, x, Z = _make_sharp_rd_data(n=800, tau=tau_true, seed=13, add_covariate=False)

    out0 = rd_estimate(y, x, c=0, p=1, covariates=Z)

    a = 0.21
    x_shift = x + a
    out1 = rd_estimate(y, x_shift, c=a, p=1, covariates=Z)

    assert out1["tau"]["bias_corrected"] == pytest.approx(out0["tau"]["bias_corrected"], abs=1e-10)
    assert out1["bandwidths"]["h_l"] == pytest.approx(out0["bandwidths"]["h_l"], abs=1e-10)
    assert out1["bandwidths"]["b_l"] == pytest.approx(out0["bandwidths"]["b_l"], abs=1e-10)


def test_rd_estimate_subset_is_ignored_by_design():
    """
    rd_estimate currently ignores the subset argument.
    This test documents that behavior explicitly.
    """
    y, x, Z = _make_sharp_rd_data(n=300, tau=1.0, seed=14)
    subset = np.arange(150, 300)

    out = rd_estimate(y, x, c=0, p=1, subset=subset)

    # rd_estimate uses the full sample
    total_used = out["N"]["left"] + out["N"]["right"]
    assert total_used == len(x)


# ============================================================
# rd_objects
# ============================================================


def test_rd_objects_basic_structure_and_bins_dataframe():
    y, x, Z = _make_sharp_rd_data(n=600, tau=1.5, seed=20)

    out = rd_objects(y, x, c=0, p=2)
    bins = out["bins"]

    # basic sanity
    assert bins.shape[0] > 0
    assert np.all(bins["N"] >= 1)
    assert np.all(bins["se"] >= 0)

    # confidence intervals are ordered
    assert np.all(bins["ci_lower"] <= bins["ci_upper"])

    # bin endpoints lie within global support
    x_min, x_max = x.min(), x.max()
    assert np.all(bins["bin_min"] >= x_min - 1e-12)
    assert np.all(bins["bin_max"] <= x_max + 1e-12)


def test_rd_objects_raises_if_c_outside_support():
    y, x, Z = _make_sharp_rd_data(n=200, tau=1.0, seed=21, add_covariate=False)

    with pytest.raises(ValueError, match="strictly inside the support"):
        rd_objects(y, x, c=5.0, p=1, covariates=Z)


def test_rd_objects_raises_if_not_enough_obs():
    y, x, Z = _make_sharp_rd_data(n=18, tau=1.0, seed=22, add_covariate=False)

    with pytest.raises(ValueError, match="Not enough observations"):
        rd_objects(y, x, c=0, p=1, covariates=Z)


def test_rd_objects_respects_bw_argument():
    y, x, Z = _make_sharp_rd_data(n=500, tau=1.0, seed=23, add_covariate=False)

    out = rd_objects(y, x, c=0, p=1, covariates=Z, bw=(0.3, 0.4))
    assert out["h"]["left"] == pytest.approx(0.3)
    assert out["h"]["right"] == pytest.approx(0.4)

    # poly grids should match those bandwidths
    assert out["poly"]["x_left"].min() == pytest.approx(-0.3)
    assert out["poly"]["x_right"].max() == pytest.approx(0.4)


def test_rd_objects_with_covariates_runs():
    y, x, Z = _make_sharp_rd_data(n=650, tau=2.0, seed=24, add_covariate=True)

    out = rd_objects(y, x, c=0, p=2, covariates=Z)

    assert len(out["coefficients"]["left"]) == 3
    assert len(out["coefficients"]["right"]) == 3
    assert out["bins"].shape[0] >= 2
