import pandas as pd

from causalopt.thresh_tune import get_rd_objects
from causalopt.utils import poly_eval


def pred_tradeoff(res: dict, inputs: pd.DataFrame = None) -> pd.DataFrame:
    """
    Objective: Constructs the welfare function based on rdrobust estimates.
    Parameters
    ----------
    results : results that come from get_rd_objects
    inputs : include if support is from another model
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
    Objective: Constructs the welfare function based on rdrobust estimates.
    Parameters
    ----------
    data : predicted values from the rdd
    outcome : name of outcome
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


def tradeoff_threshold(
    df: pd.DataFrame,
    outcomes: list,
    prob_col: str,
    threshold: float,
):
    """
    Objective: This function performs the overall threshold tuning. It returns the optimum threshold and the expected gain
    in the outcome variable.
    ----------
    The output of this function is the optimum threshold and the expected gain in the outcome variable resulting from the tuning.

    Parameters
    ----------
    df : DataFrame
           Original data.
    outcomes: list
        List of outcome variables.
    prob_col : str
           Column name of the running variable or probability from an ML model.
    threshold : str
           Current threshold. We will center the running variable at this point.
    """

    y = df[outcomes[0]]
    x = df[prob_col] - threshold
    c = 0

    rdobj_main = get_rd_objects(y, x, c)

    pred_main = pred_tradeoff(rdobj_main, inputs=None)

    w_main = gains_eval(pred_main, outcomes[0])

    final_result = w_main

    for out in outcomes[1:]:
        y = df[out]

        rdobj_alt = get_rd_objects(y, x, c)

        pred_alt = pred_tradeoff(rdobj_alt, pred_main[["x", "n", "p"]])

        w_alt = gains_eval(pred_alt, out)

        final_result = pd.merge(final_result, w_alt, on=["x", "n", "p"], how="inner")

    return final_result
