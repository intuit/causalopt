import pandas as pd

from causalopt.utils import poly_eval


def pred_tradeoff(res: dict, inputs: pd.DataFrame = None) -> pd.DataFrame:
    """
    Evaluate left- and right-side polynomial predictions at a given support.

    If inputs is None, the support is taken from the bin midpoints in res.
    Otherwise, x, n, and p are read from inputs, which allows evaluating the
    polynomial of one outcome on the support of another.

    Parameters
    ----------
    res : dict
        Output of get_rd_objects.
    inputs : pd.DataFrame or None
        Optional DataFrame with columns x, n, p to override the default support.
    """

    if inputs is None:
        bins = res["bins"]  # pandas.DataFrame
        x = bins["mean_x"].to_numpy()
        n = sum(bins["N"].to_numpy())
        p = bins["N"].to_numpy() / n
    else:
        x = inputs["x"].to_numpy()
        n = inputs["n"].to_numpy()
        p = inputs["p"].to_numpy()

    b_l = res["b_l"]
    b_r = res["b_r"]

    y_hat_l = poly_eval(x, b_l)
    y_hat_r = poly_eval(x, b_r)

    data_predictions = pd.DataFrame(
        {"x": x, "n": n, "p": p, "y_hat_l": y_hat_l, "y_hat_r": y_hat_r}
    )

    return data_predictions


def gains_eval(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Compute the cumulative welfare gain relative to the current threshold.

    Calculates per-bin weighted welfare (treatment effect × proportion), accumulates
    it from the highest x downward, then computes the gain over the current threshold
    welfare and scales it by total n.

    Parameters
    ----------
    data : pd.DataFrame
        Output of pred_tradeoff.
    outcome : str
        Name of the outcome variable (used to name the gain column).
    """
    gain_outcome = f"gain_{outcome}"

    data["w_outcome"] = (data["y_hat_r"] - data["y_hat_l"]) * data["p"]
    data = data.sort_values(by="x", ascending=False)
    data["welfare_outcome"] = data["w_outcome"].cumsum()

    current_threshold_key = data["x"].abs().idxmin()  # Current Threshold
    current_welfare_optimum = data.loc[current_threshold_key, "welfare_outcome"]

    data[gain_outcome] = data["n"] * (data["welfare_outcome"] - current_welfare_optimum)

    data_welfare = data[["x", "n", "p", gain_outcome]]

    return data_welfare
