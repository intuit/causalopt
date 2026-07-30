# CausalOpt - Causal Optimizer

[![PyPI version](https://img.shields.io/pypi/v/causalopt.svg)](https://pypi.org/project/causalopt/)
![Supported Python versions](https://img.shields.io/badge/python-3.11_|_3.12_|_3.13-green?logo=python)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**CausalOpt** is a Python package for causal inference using Regression Discontinuity Design (RDD). It provides tools for estimating treatment effects at thresholds, optimizing decision thresholds, and visualizing RDD results — for both a single **binary** cutoff and a **multiclass** decision.

A single entry point, `causalopt(..., mode="binary" | "multiclass")`, dispatches to the right tuner and returns a standardized result. Lower-level functions (`rd_estimate`, `bw_select`, `optimum_threshold`, `tradeoff_threshold`, `get_thresholds`) remain available for finer control.

## Features

- **RDD Estimation**: Estimate sharp regression discontinuity treatment effects using local polynomial regression with robust bias correction (Calonico-Cattaneo-Titiunik, 2014)
- **Bandwidth Selection**: MSE-optimal bandwidth selection with triangular kernel and nearest-neighbor variance estimation
- **Binary Threshold Optimization**: Find optimal thresholds that maximize welfare gains based on RDD estimates
- **Multiclass Threshold Tuning**: Tune the decision boundaries of a K-class argmax rule `T = argmax_k (p_k - τ_k)`, on the probability simplex or on arbitrary scores
- **Multi-Outcome Tradeoffs**: Analyze threshold tradeoffs across multiple outcome variables on a shared grid
- **Unified Entry Point**: One `causalopt(mode=...)` call for both the binary and multiclass workflows
- **Scalability & Extrapolation**: Optional binning of large datasets and an `out_of_bandwidth` option to report tradeoffs over the full support
- **Visualization**: Publication-ready plots for RDD impact, threshold recommendations, and (K=3) simplex decision regions

## Installation

### Using pip

```bash
pip install causalopt
```

### Using Poetry

```bash
poetry add causalopt
```

## Quickstart

### Basic RDD Estimation

```python
import pandas as pd
from causalopt.estimation import rd_estimate, bw_select

# Load your data
df = pd.read_csv("your_data.csv")

# Define outcome and running variable
y = df["outcome"]
x = df["running_variable"]
cutoff = 0.5  # Your threshold/cutoff value

# Select optimal bandwidth
bw = bw_select(y=y, x=x, c=cutoff)
print(f"Optimal bandwidth: {bw['bandwidths']}")

# Estimate RDD treatment effect
results = rd_estimate(y=y, x=x, c=cutoff)

print(f"Treatment effect: {results['tau']['bias_corrected']:.4f}")
print(f"95% CI: [{results['ci']['robust'][0]:.4f}, {results['ci']['robust'][1]:.4f}]")
```

### Binary Threshold Tuning (unified entry point)

Use `causalopt` with `mode="binary"` to tune a single cutoff. The first outcome is the *primary* outcome that drives the recommendation; any additional outcomes are carried along and evaluated on the **same** grid, so the recommendation and the tradeoff curve are directly comparable.

```python
from causalopt import causalopt

results = causalopt(
    df,
    outcomes=["primary_outcome", "secondary_outcome"],  # first is primary
    score_cols="probability_score",                     # the running variable
    mode="binary",
    threshold=0.5,                                       # current cutoff
)

# Recommendation for the primary outcome
opt = results["optimum"]
print(f"Recommendation: {opt['Recommendation']}")
print(f"Optimum threshold: {opt['Thresholds']['Optimum']:.4f}")
print(f"Conservative threshold: {opt['Thresholds']['Conservative']:.4f}")
print(f"Expected welfare gain: {opt['Additional Gain']['Optimum']:.2f}")

# Tradeoff frontier: gain_<outcome> per candidate threshold (x = 0 is current)
frontier = results["frontier"]
print(frontier.head())
```

The result is a dict `{"mode", "current", "frontier", "optimum", "details"}`, where `details` carries the underlying RD estimates and prediction objects used by the plots.

> The lower-level `optimum_threshold` and `tradeoff_threshold` functions are still available (`from causalopt import optimum_threshold, tradeoff_threshold`) if you want to run just one analysis.

### Multiclass Threshold Tuning

Use `causalopt` with `mode="multiclass"` to tune the boundaries of a K-class argmax rule. Pass the `K` probability/score columns in `score_cols` and the current threshold vector `tau`.

```python
from causalopt import causalopt

results = causalopt(
    df,
    outcomes=["Y", "R"],                        # one or more outcomes
    score_cols=["Prob_1", "Prob_2", "Prob_3"],  # K class scores
    mode="multiclass",
    tau=[0.33, 0.33, 0.34],                     # current thresholds
    probabilities=True,                          # simplex inputs (default)
)

# Welfare-optimal tau per outcome, with the other outcomes at that tau
for outcome, rec in results["optimum"].items():
    print(outcome, "->", tuple(round(t, 3) for t in rec["tau"]),
          "mean gain", round(rec["mean"], 4))

# Full candidate frontier: tau_1..K plus mean_/total_<outcome>
print(results["frontier"].head())
```

Set `probabilities=False` to treat the columns as arbitrary real-valued scores (no simplex constraint); tune `B` (search budget) and `kernel` (how per-boundary weights combine) as needed.

### Visualization

```python
from causalopt import rdd_impact, plot_thresh

# rdd_impact / plot_thresh accept the unified binary `causalopt` result
# (or a legacy `optimum_threshold` result) directly.
impact_plot = rdd_impact(results, outcome_col="primary_outcome")
impact_plot.save("rdd_impact.png")

thresh_plot = plot_thresh(results, outcome_col="primary_outcome")
thresh_plot.save("threshold_recommendations.png")
```

For the multiclass case with `K = 3`, plot the decision regions on the simplex:

```python
from causalopt.multiclass import decision_rule, plot_simplex

tau_opt = results["optimum"]["Y"]["tau"]
df["T"] = decision_rule(df[["Prob_1", "Prob_2", "Prob_3"]].values, tau_opt)
plot_simplex(df, tau_opt)
```

### Example notebooks

End-to-end, runnable walkthroughs live in [`examples/`](examples/):

- [`examples/binary_case.ipynb`](examples/binary_case.ipynb) — the binary workflow on simulated data.
- [`examples/multiclass_case.ipynb`](examples/multiclass_case.ipynb) — the multiclass (K=3, simplex) workflow.

## Use Cases

CausalOpt is designed for scenarios where you need to:

1. **Evaluate ML model thresholds**: Assess the causal impact of decisions made at probability thresholds from machine learning models
2. **Optimize decision boundaries**: Find the threshold that maximizes a desired outcome while accounting for uncertainty
3. **Policy evaluation**: Estimate treatment effects in settings with sharp cutoffs (e.g., eligibility thresholds, scoring systems)
4. **A/B test alternatives**: When randomized experiments aren't feasible, use RDD to estimate causal effects from observational data with natural thresholds
