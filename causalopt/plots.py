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


def rdd_impact(results, outcome_col: str):
    """
     Objective: Graph thresholds
     ----------
     rdd_results : result output from rdrobust
     rdd_predictions : results that come from rdd_predictions
     df : DataFrame
       Original data.
    outcome_col : str
           Name of the outcome variable.
    """
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


def plot_thresh(results, outcome_col: str):
    """
    Objective: Graph thresholds
    ----------
    results: results from optimum_threshold
    outcome_col: outcome variable
    """

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
