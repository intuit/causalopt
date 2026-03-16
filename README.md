# CausalOpt

[![PyPI version](https://img.shields.io/pypi/v/causalopt.svg)](https://pypi.org/project/causalopt/)
![Supported Python versions](https://img.shields.io/badge/python-3.11_|_3.12_|_3.13-green?logo=python)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**CausalOpt** is a Python package for causal inference using Regression Discontinuity Design (RDD). It provides tools for estimating treatment effects at thresholds, optimizing decision thresholds, and visualizing RDD results.

## Features

- **RDD Estimation**: Estimate sharp regression discontinuity treatment effects using local polynomial regression with robust bias correction (Calonico-Cattaneo-Titiunik, 2014)
- **Bandwidth Selection**: MSE-optimal bandwidth selection with triangular kernel and nearest-neighbor variance estimation
- **Threshold Optimization**: Find optimal thresholds that maximize welfare gains based on RDD estimates
- **Multi-Outcome Tradeoffs**: Analyze threshold tradeoffs across multiple outcome variables
- **Visualization**: Publication-ready plots for RDD impact and threshold recommendations

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

### Threshold Optimization

Find the optimal threshold that maximizes welfare:

```python
from causalopt.thresh_tune import exc_optim_thresh

# Run threshold optimization
results = exc_optim_thresh(
    df=df,
    outcome_col="outcome",
    prob_col="probability_score",
    threshold=0.5,  # Current threshold
)

# Get recommended thresholds
thresholds = results["optimum_thresholds"]
print(f"Recommendation: {thresholds['Recommendation']}")
print(f"Optimum threshold: {thresholds['Thresholds']['Optimum']:.4f}")
print(f"Conservative threshold: {thresholds['Thresholds']['Conservative']:.4f}")
print(f"Expected welfare gain: {thresholds['Additional Gain']['Optimum']:.2f}")
```

### Multi-Outcome Tradeoff Analysis

Analyze threshold tradeoffs when you have multiple outcomes:

```python
from causalopt.thresh_tradeoff import tradeoff_threshold

# Analyze tradeoffs across multiple outcomes
tradeoff_results = tradeoff_threshold(
    df=df,
    outcomes=["primary_outcome", "secondary_outcome", "cost"],
    prob_col="probability_score",
    threshold=0.5,
)

print(tradeoff_results)
```

### Visualization

```python
from causalopt.plots import rdd_impact, plot_thresh

# Plot RDD impact
impact_plot = rdd_impact(results, outcome_col="outcome")
impact_plot.save("rdd_impact.png")

# Plot threshold recommendations
thresh_plot = plot_thresh(results, outcome_col="outcome")
thresh_plot.save("threshold_recommendations.png")
```

## Use Cases

CausalOpt is designed for scenarios where you need to:

1. **Evaluate ML model thresholds**: Assess the causal impact of decisions made at probability thresholds from machine learning models
2. **Optimize decision boundaries**: Find the threshold that maximizes a desired outcome while accounting for uncertainty
3. **Policy evaluation**: Estimate treatment effects in settings with sharp cutoffs (e.g., eligibility thresholds, scoring systems)
4. **A/B test alternatives**: When randomized experiments aren't feasible, use RDD to estimate causal effects from observational data with natural thresholds
