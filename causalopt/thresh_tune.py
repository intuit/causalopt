import numpy as np
import pandas as pd

from causalopt.estimation import (
    rd_estimate,
    rd_objects,
)
from causalopt.utils import poly_eval, sim_poly_ic


def get_rd_objects(y, x, c):
    rdest = rd_estimate(y=y, x=x, c=c)
    h_l = rdest["bandwidths"]["h_l"]
    h_r = rdest["bandwidths"]["h_r"]
    bw = [h_l, h_r]

    subset = (-h_l <= x) & (x <= h_r)

    rdobj = rd_objects(y=y, x=x, c=c, bw=bw, subset=subset)

    b_l = rdest["coef_poly"]["bias_corrected"]["left"]
    b_r = rdest["coef_poly"]["bias_corrected"]["right"]
    v_l = rdest["V_poly"]["bias_corrected"]["left"]
    v_r = rdest["V_poly"]["bias_corrected"]["right"]

    bins = rdobj["bins"]

    point_estimate = rdest["tau"]["bias_corrected"]
    point_estimate_lb = rdest["ci"]["bias_corrected"][0]
    point_estimate_ub = rdest["ci"]["bias_corrected"][1]

    estimates = [point_estimate, point_estimate_lb, point_estimate_ub]

    return {
        "rd_estimates": rdest,
        "rd_plot_objects": rdobj,
        "b_l": b_l,
        "b_r": b_r,
        "v_l": v_l,
        "v_r": v_r,
        "bins": bins,
        "estimates": estimates,
    }


def predictions(rdres: dict):
    """
    Objective: Constructs the welfare function based on rdrobust estimates.
    Parameters
    ----------
    rdres : results that come from get_rd_objects
    """

    b_l = rdres["b_l"]
    b_r = rdres["b_r"]
    v_l = rdres["v_l"]
    v_r = rdres["v_r"]
    bins = rdres["bins"]

    x = bins["mean_x"].to_numpy()
    n = sum(bins["N"].to_numpy())
    p = bins["N"].to_numpy() / n

    y_mean = bins["mean_y"]
    y_ci_l = bins["ci_lower"]
    y_ci_r = bins["ci_upper"]

    y_hat_l = poly_eval(x, b_l)
    y_hat_r = poly_eval(x, b_r)

    y_fit_l, y_hat_lower_l, y_hat_upper_l = sim_poly_ic(
        x, b_l, v_l, nsim=10000, alpha=0.05, seed=None
    )
    y_fit_r, y_hat_lower_r, y_hat_upper_r = sim_poly_ic(
        x, b_r, v_r, nsim=10000, alpha=0.05, seed=None
    )

    data_predictions = pd.DataFrame(
        {
            "x": x,
            "n": n,
            "p": p,
            "y_mean": y_mean,
            "y_ci_l": y_ci_l,
            "y_ci_r": y_ci_r,
            "y_hat_l": y_hat_l,
            "y_hat_lower_l": y_hat_lower_l,
            "y_hat_upper_l": y_hat_upper_l,
            "y_hat_r": y_hat_r,
            "y_hat_lower_r": y_hat_lower_r,
            "y_hat_upper_r": y_hat_upper_r,
        }
    )

    return data_predictions


def welfare(
    data: pd.DataFrame,
    rdres: dict,
):
    """
    Objective: Constructs the welfare function based on rdrobust estimates.
    Parameters
    ----------
    data : predicted values from the rdd
    rdres : rdd estimate from get_rd_objects
    """

    estimate = rdres["estimates"][0]

    if estimate >= 0:
        data["w_optimum"] = (data["y_hat_r"] - data["y_hat_l"]) * data["p"]
        data["w_conservative"] = (data["y_hat_lower_r"] - data["y_hat_l"]) * data["p"]
        data["w_aggressive"] = (data["y_hat_upper_r"] - data["y_hat_l"]) * data["p"]

    else:
        data["w_optimum"] = (data["y_hat_r"] - data["y_hat_l"]) * data["p"]
        data["w_conservative"] = (data["y_hat_r"] - data["y_hat_lower_l"]) * data["p"]
        data["w_aggressive"] = (data["y_hat_r"] - data["y_hat_upper_l"]) * data["p"]

    data = data.sort_values(by="x", ascending=False)
    data["welfare_optimum"] = data["w_optimum"].cumsum()
    data["welfare_conservative"] = data["w_conservative"].cumsum()
    data["welfare_aggressive"] = data["w_aggressive"].cumsum()

    data_welfare = data[
        [
            "x",
            "n",
            "p",
            "w_optimum",
            "w_conservative",
            "w_aggressive",
            "welfare_optimum",
            "welfare_conservative",
            "welfare_aggressive",
        ]
    ]

    return data_welfare


def optim_thresh(data_welfare: pd.DataFrame, rdres: dict, current_threshold):
    """
    Objective: finding optimum threshold using the function welfare_function and current threshold.

    Parameters
    ----------
    data_welfare : Welfare function results from welfare_function
    estimates: point estimate, lb and ub from get_rd_objects
    """

    estimates = rdres["estimates"]

    if estimates[0] >= 0:
        data_welfare = data_welfare.loc[data_welfare["x"] <= 0]
    else:
        data_welfare = data_welfare.loc[data_welfare["x"] >= 0]

    max_welfare_optimum = data_welfare["welfare_optimum"].max()  # numerical maximum
    max_welfare_optimum_key = data_welfare[
        "welfare_optimum"
    ].idxmax()  # row label (index) where that max sits
    max_welfare_conservative = data_welfare["welfare_conservative"].max()  # numerical maximum
    max_welfare_conservative_key = data_welfare[
        "welfare_conservative"
    ].idxmax()  # row label (index) where that max sits
    max_welfare_aggressive = data_welfare["welfare_aggressive"].max()  # numerical maximum
    max_welfare_aggressive_key = data_welfare[
        "welfare_aggressive"
    ].idxmax()  # row label (index) where that max sits

    optimum_threshold = (
        data_welfare.loc[max_welfare_optimum_key, "x"] + current_threshold
    )  # Optimum Threshold
    conservative_threshold = (
        data_welfare.loc[max_welfare_conservative_key, "x"] + current_threshold
    )  # Conservative Threshold
    aggressive_threshold = (
        data_welfare.loc[max_welfare_aggressive_key, "x"] + current_threshold
    )  # Aggressive Threshold

    current_threshold_key = data_welfare["x"].abs().idxmin()  # Current Threshold
    current_welfare_optimum = data_welfare.loc[current_threshold_key, "welfare_optimum"]
    current_welfare_conservative = data_welfare.loc[current_threshold_key, "welfare_conservative"]
    current_welfare_aggressive = data_welfare.loc[current_threshold_key, "welfare_aggressive"]
    n = data_welfare["n"].max()

    welfare_gain_optimum = n * (max_welfare_optimum - current_welfare_optimum)
    welfare_gain_conservative = n * (max_welfare_conservative - current_welfare_conservative)
    welfare_gain_aggressive = n * (max_welfare_aggressive - current_welfare_aggressive)

    if np.sign(estimates[0]) == 1 and np.sign(estimates[1]) == np.sign(estimates[2]):
        recommendation = "Estimates are POSITIVE we should REDUCE the threshold"
    if np.sign(estimates[0]) == -1 and np.sign(estimates[1]) == np.sign(estimates[2]):
        recommendation = "Estimates are NEGATIVE we should INCREASE the threshold"
    if np.sign(estimates[1]) != np.sign(estimates[2]):
        recommendation = "Estimates are Not Stat. Sig. we should KEEP the current threshold"

    result = {
        "Recommendation": recommendation,
        "Thresholds": {
            "Optimum": optimum_threshold,
            "Conservative": conservative_threshold,
            "Aggressive": aggressive_threshold,
        },
        "Additional Gain": {
            "Optimum": welfare_gain_optimum,
            "Conservative": welfare_gain_conservative,
            "Aggressive": welfare_gain_aggressive,
        },
    }

    return result


def exc_optim_thresh(df: pd.DataFrame, outcome_col: str, prob_col: str, threshold: float):
    """
    Objective: This function performs the overall threshold tuning. It returns the optimum threshold and the expected gain
    in the outcome variable.
    ----------
    The output of this function is the optimum threshold and the expected gain in the outcome variable resulting from the tuning.

    Parameters
    ----------
    df : DataFrame
           Original data.
    outcome_col : str
           Column name of the outcome variable.
    prob_col : str
           Column name of the running variable or probability from an ML model.
    threshold : float
           Current threshold. We will center the running variable at this point.
    """

    data_descriptives = pd.DataFrame(df.describe())

    y = df[outcome_col]
    x = df[prob_col] - threshold
    c = 0

    rd_obj = get_rd_objects(y, x, c)

    rd_predictions = predictions(rd_obj)

    welfare_results = welfare(rd_predictions, rd_obj)

    optimum_thresholds = optim_thresh(welfare_results, rd_obj, threshold)

    final_result = {
        "data_descriptives": data_descriptives,
        "rd_results": rd_obj["rd_estimates"],
        "rdplot": rd_obj["rd_plot_objects"],
        "predictions": rd_predictions,
        "welfare": welfare_results,
        "optimum_thresholds": optimum_thresholds,
        "current_threshold": threshold,
    }

    return final_result
