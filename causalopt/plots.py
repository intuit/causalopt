from plotnine import (
    aes,
    coord_cartesian,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_vline,
    ggplot,
    labs,
    theme,
    theme_bw,
)


def _normalize_results(results: dict) -> dict:
    """
    Accept either result shape used by the plot helpers.

    The plot helpers were written against the flat ``optimum_threshold`` output
    (keys ``rd_results``, ``predictions``, ``optimum_thresholds``,
    ``current_threshold``, ``data_descriptives``). The unified ``causalopt``
    entry point (``mode="binary"``) instead nests those objects under
    ``details`` and renames a couple of keys. This helper detects the unified
    shape and remaps it to the flat keys so both can be plotted with the same
    functions. A flat result is returned unchanged.
    """
    if "details" in results and "optimum" in results and "frontier" in results:
        flat = dict(results["details"])
        flat["optimum_thresholds"] = results["optimum"]
        current = results.get("current") or {}
        flat["current_threshold"] = current.get("threshold")
        return flat
    return results


def rdd_impact(results: dict, outcome_col: str):
    """
    Plot the RD impact at the current threshold for a single outcome variable.

    Produces a ggplot showing:
        - Bin means as scatter points (effectively invisible; used to set the
          axis scale from the observed data range)
        - Fitted left polynomial (red line) for observations below the cutoff
        - Fitted right polynomial (blue line) for observations above the cutoff
        - A grey confidence ribbon across the full x range (95% simulation CI)
        - A vertical dashed line at x=0 (the centered threshold)

    The y-axis is clipped to the overlap between the observed data range and the
    prediction CI range to avoid excessive whitespace.

    The plot caption reports:
        - Bias-corrected point estimate of the RD treatment effect
        - 95% bias-corrected confidence interval
        - ITC (implied treatment-to-control ratio): the percent change in the
          outcome at the threshold relative to the control mean

    Parameters
    ----------
    results : dict
        Output of optimum_threshold or the unified causalopt (mode="binary")
        entry point. Must expose "data_descriptives", "rd_results", and
        "predictions" (directly or nested under "details").
    outcome_col : str
        Name of the outcome variable. Used in the plot title, y-axis label,
        and to look up the observed min/max from data_descriptives.

    Returns
    -------
    plotnine.ggplot
        A ggplot object. Call .show() or display inline in a notebook.
    """
    results = _normalize_results(results)

    # colors
    blue = "#0177c9"
    red = "#bd0707"

    #
    data_descriptives = results["data_descriptives"]
    rdest = results["rd_results"]
    predictions = results["predictions"]

    point_estimate = rdest["tau"]["bias_corrected"]
    point_estimate_lb = rdest["ci"]["bias_corrected"][0]
    point_estimate_ub = rdest["ci"]["bias_corrected"][1]

    # Bounds for the vertical axis
    ymin_d = data_descriptives.loc["min", outcome_col]
    ymax_d = data_descriptives.loc["max", outcome_col]

    ymin_e = predictions["y_ci_l"].min()
    ymax_e = predictions["y_ci_r"].max()

    ymin = max(ymin_d, ymin_e)
    ymax = min(ymax_d, ymax_e)

    avcontrol = sum(
        predictions["n"][predictions["x"] < 0] * predictions["y_mean"][predictions["x"] < 0]
    ) / sum(predictions["n"][predictions["x"] < 0])
    itc = round(100 * (point_estimate + avcontrol) / avcontrol, 1)

    graph = (
        ggplot()
        + geom_point(aes(x="x", y="y_mean"), data=predictions, size=-0.1)
        + geom_line(
            aes(x="x", y="y_hat_l"),
            data=predictions[(predictions["x"] <= 0)],
            color=red,
        )
        + geom_line(
            aes(x="x", y="y_hat_r"),
            data=predictions[(predictions["x"] >= 0)],
            color=blue,
        )
        + geom_ribbon(
            aes(x="x", ymin="y_ci_l", ymax="y_ci_r"),
            data=predictions,
            fill="grey",
            alpha=0.3,
        )
        + geom_vline(xintercept=0, linetype="dashed", size=0.5)
        + labs(title=f"Impact on {outcome_col}")
        + labs(y=f"Average {outcome_col}", x="Centered Probability (Threshold=0)")
        + labs(
            caption=f"Impact at Threshold =  {point_estimate:,.2f}, IC95%[{point_estimate_lb:,.2f},{point_estimate_ub:,.2f}], ITC = {itc:,.1f}"
        )
        + coord_cartesian(ylim=(ymin, ymax))
        + theme_bw()
        + theme(legend_position="none")
    )

    return graph


def plot_thresh(results: dict, outcome_col: str):
    """
    Plot RD polynomial predictions with candidate threshold lines.

    Visualises where the threshold should move and by how much, using the
    welfare-maximizing thresholds from optimum_threshold. The x-axis is in the
    original (uncentered) scale of the running variable.

    Visual elements:
        - Solid left polynomial (red) for x ≤ current_threshold
        - Solid right polynomial (blue) for x ≥ current_threshold
        - Dashed extension of the active-side polynomial across the other side,
          showing the counterfactual outcome if the threshold were moved there.
          The active side is determined by the sign of the RD estimate:
            positive estimate → right-side polynomial extended left (dashed blue)
            negative estimate → left-side polynomial extended right (dashed red)
        - Grey confidence ribbon for the active-side polynomial across all x
        - Four vertical dashed lines:
            black  – current threshold
            blue   – optimum threshold (maximises expected welfare gain)
            red    – conservative threshold (uses lower CI of active polynomial)
            green  – aggressive threshold (uses upper CI of active polynomial)

    Parameters
    ----------
    results : dict
        Output of optimum_threshold or the unified causalopt (mode="binary")
        entry point. Must expose "rd_results", "predictions",
        "optimum_thresholds", and "current_threshold" (directly or nested under
        "details" / "optimum" / "current").
    outcome_col : str
        Name of the outcome variable. Used in the plot title and y-axis label.

    Returns
    -------
    plotnine.ggplot
        A ggplot object. Call .show() or display inline in a notebook.
    """
    results = _normalize_results(results)

    rdest = results["rd_results"]
    predictions = results["predictions"]
    optimum_thresholds = results["optimum_thresholds"]
    current_threshold = results["current_threshold"]

    point_estimate = rdest["tau"]["bias_corrected"]

    # palette
    blue = "#0177c9"
    green = "#54b709"
    red = "#bd0707"

    optimum_threshold = optimum_thresholds["Thresholds"]["Optimum"]
    conservative_threshold = optimum_thresholds["Thresholds"]["Conservative"]
    aggressive_threshold = optimum_thresholds["Thresholds"]["Aggressive"]
    dfgraph = predictions.copy()
    dfgraph["x"] = dfgraph["x"] + current_threshold

    if point_estimate >= 0:
        graph = (
            ggplot()
            + geom_line(
                aes("x", "y_hat_l"),
                data=dfgraph[(dfgraph["x"] <= current_threshold)],
                colour=red,
                size=1,
            )
            + geom_line(
                aes("x", "y_hat_r"),
                data=dfgraph[(dfgraph["x"] >= current_threshold)],
                colour=blue,
                size=1,
            )
            + geom_line(
                aes("x", "y_hat_r"),
                data=dfgraph[(dfgraph["x"] <= current_threshold)],
                colour=blue,
                size=1,
                linetype="dashed",
            )
            + geom_ribbon(
                aes(x="x", ymin="y_hat_lower_r", ymax="y_hat_upper_r"),
                data=dfgraph,
                fill="grey",
                alpha=0.3,
            )
            + geom_vline(xintercept=current_threshold, linetype="dashed", size=0.5)
            + geom_vline(
                xintercept=optimum_threshold,
                color=blue,
                linetype="dashed",
                size=0.5,
            )
            + geom_vline(
                xintercept=conservative_threshold,
                color=red,
                linetype="dashed",
                size=0.5,
            )
            + geom_vline(
                xintercept=aggressive_threshold,
                color=green,
                linetype="dashed",
                size=0.5,
            )
            + labs(title=f"RDD Predictions and Confidence Intervals - {outcome_col}")
            + labs(y=f"Average {outcome_col}", x="Probability")
            + labs(
                caption="Thresholds (Vertical lines): Black = Current, Blue = Optimum, Red = Conservative, Green = Aggressive"
            )
            + theme_bw()
            + theme(legend_position="none")
        )
    else:
        graph = (
            ggplot()
            + geom_line(
                aes("x", "y_hat_r"),
                data=dfgraph[(dfgraph["x"] >= current_threshold)],
                colour=blue,
                size=1,
            )
            + geom_line(
                aes("x", "y_hat_l"),
                data=dfgraph[(dfgraph["x"] <= current_threshold)],
                colour=red,
                size=1,
            )
            + geom_line(
                aes("x", "y_hat_l"),
                data=dfgraph[(dfgraph["x"] >= current_threshold)],
                colour=red,
                size=1,
                linetype="dashed",
            )
            + geom_ribbon(
                aes(x="x", ymin="y_hat_lower_l", ymax="y_hat_upper_l"),
                data=dfgraph,
                fill="grey",
                alpha=0.3,
            )
            + geom_vline(xintercept=current_threshold, linetype="dashed", size=0.5)
            + geom_vline(
                xintercept=optimum_threshold,
                color=blue,
                linetype="dashed",
                size=0.5,
            )
            + geom_vline(
                xintercept=conservative_threshold,
                color=red,
                linetype="dashed",
                size=0.5,
            )
            + geom_vline(
                xintercept=aggressive_threshold,
                color=red,
                linetype="dashed",
                size=0.5,
            )
            + labs(title=f"RDD Predictions and Confidence Intervals - {outcome_col}")
            + labs(y=f"Average {outcome_col}", x="Probability")
            + labs(
                caption="Thresholds (Vertical lines): Black = Current, Blue = Optimum, Red = Conservative, Green = Aggressive"
            )
            + theme_bw()
            + theme(legend_position="none")
        )

    return graph
