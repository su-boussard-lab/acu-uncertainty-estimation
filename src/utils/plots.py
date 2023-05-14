import os
from typing import Dict, List
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import calibration as cal
import arviz as az
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score, recall_score, precision_score

from src.utils.metrics import net_benefit_curve, treat_all_curve, treat_none_curve

plt.rcParams["axes.facecolor"] = "w"
plt.rcParams["figure.facecolor"] = "w"
plt.style.use("seaborn-colorblind")

import cycler

jama_style_colors = [
    "#374E55FF",
    "#DF8F44FF",
    "#00A1D5FF",
    "#B24745FF",
    "#79AF97FF",
    "#6A6599FF",
    "#80796BFF",
]

plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", jama_style_colors)


def calibration_plot(preds: Dict, y_true: np.ndarray, save_path: str) -> None:
    """Creates calibration plot for a list of models and their names
    Args:
        preds (Dict): dictionary of predictions with model name as key and risk predictions of the test set as value
        y_true (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=1, figsize=(8, 6))

    # Plot perfectly calibrated
    ax = axes
    for model, y_pred in preds.items():
        x, y = calibration_curve(y_true, y_pred, n_bins=20)

        # Plot model's calibration curve
        ax.plot(y, x, marker=".", label=model)

    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideally Calibrated")
    # ax.set_title(f"Calibration Curve")
    ax.legend()
    ax.set_xlabel("Average Predicted Probability in each bin")
    ax.set_ylabel("Ratio of positives")
    # ax.grid()
    plt.savefig(
        os.path.join(save_path, "calibrations.pdf"), dpi=300.0, bbox_inches="tight"
    )


def violin_plot(metric: str, preds: Dict, save_path: str) -> None:
    """Creates a violin plot of the bootstrapped metrics
    Args:
        metric (str): name of the metric that is compared
        preds (Dict): dictionary of predictions with model name as key and bootstraps as values
        save_path (str): path where the figure will be stored
    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=1, figsize=(9, 3))

    data_df = pd.DataFrame(preds)
    sns.violinplot(
        data=data_df,
    )

    # plt.title(f"{metric} Bootstrapped Scores")
    plt.ylabel(f"{metric}")
    # #plt.grid()
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.savefig(
        os.path.join(save_path, f"{metric}_bootstrap_violin.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def net_benefit_plot(preds: Dict, y_true: np.ndarray, save_path: str) -> None:
    """Creates net benefit plot for a list of models and their names
    Args:
        preds (Dict): dictionary of predictions with model name as key and risk predictions of the test set as value
        y_true (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=1, figsize=(9, 3))

    ax = axes
    for model, y_pred in preds.items():
        net_benefit_curve(ax, y_true, y_pred, title=model)
    treat_all_curve(ax, y_true, lowest_net_benefit=-0.2, title="Treat All")
    treat_none_curve(ax, title="Treat None")

    # ax.set_title(f"Net Benefit")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=-0.1)
    ax.set_xlabel("Risk Threshold")
    ax.set_ylabel("Net Benefit Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.grid()
    plt.savefig(
        os.path.join(save_path, "net_benefit.pdf"), dpi=300.0, bbox_inches="tight"
    )


def uncertainty_vs_predictions(preds: Dict, save_path: str) -> None:
    """Creates scatter plots of predictions and uncertainties
    Args:
        preds (Dict): dictionary of predictive distributions with model name as key and their distributions of the test set as value
        save_path (str): path where the figure will be stored
    Returns:
        None
    """
    preds.pop("Frequentist LASSO")

    fig, axes = plt.subplots(ncols=1, figsize=(6, 4.5))
    # Plot perfectly calibrated
    ax = axes
    from scipy.special import entr

    for model, pred_distribution in preds.items():
        y_pred = pred_distribution.mean(0)
        y_uncerts = pred_distribution.std(0)
        # y_uncerts = entr(pred_distribution).mean(axis=0)
        # Plot model's calibration curve
        ax.scatter(y_pred, y_uncerts, marker=".", label=model, s=2, alpha=0.5)

    # ax.set_title(f"Uncertainties at Risk Predictions")
    ax.legend(loc="upper right", markerscale=5)
    ax.set_xlabel("Predicted Risk Probability ($\\bar{y}$)")
    ax.set_ylabel("Predictive Uncertainty ($\sigma$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.grid()
    plt.savefig(
        os.path.join(save_path, "uncertainty_scatter_plot.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def single_predictions(
    preds: Dict,
    y_test: np.ndarray,
    save_path: str,
    plot_uncertainty_boarders: bool = False,
) -> None:
    """Creates plots of the predictive distribution for 3 patients
    Args:
        preds (Dict): dictionary of predictive distributions with model name as key and their distributions of the test set as value
        y_test (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
        plot_uncertainty_boarders (bool, False): if true, the uncertainty boarders are also plotted
    Returns:
        None
    """
    titles = ["a", "b", "c"]
    interesting_index = [1657, 1, 1259]

    # fig, axes = plt.subplots(ncols=3, figsize=(24, 6))

    lasso_preds = preds.pop("Frequentist LASSO")

    y_bar_string = "$\\bar{y}$"
    for i, (title, pat_index) in enumerate(zip(titles, interesting_index)):

        figure(figsize=(5, 5))
        cols = iter(jama_style_colors)
        y_true = y_test.values[pat_index]

        cols = iter(jama_style_colors)
        for model, pred_distribution in preds.items():
            current_col = next(cols)
            distribution_of_patient = pred_distribution[:, pat_index]
            y_pred = distribution_of_patient.mean()
            plt.axvline(
                x=y_pred, color=current_col, label=f"{y_bar_string} {model}", alpha=0.7
            )
            # Plot uncertainty bounds
            if plot_uncertainty_boarders:
                y_uncert = distribution_of_patient.std()
                plt.axvline(x=y_pred - y_uncert, color=current_col, linestyle="--")
                plt.axvline(x=y_pred + y_uncert, color=current_col, linestyle="--")

        # frequentist LASSO
        plt.axvline(
            x=lasso_preds[pat_index],
            color=next(cols),
            label="$\\bar{y}$ Frequentist LASSO",
            alpha=0.7,
        )

        cols = iter(jama_style_colors)
        for model, pred_distribution in preds.items():
            distribution_of_patient = pred_distribution[:, pat_index]

            current_col = next(cols)
            sns.kdeplot(
                distribution_of_patient,
                # bins=20,
                label=model,
                # ax=ax,
                # stat="probability",
                # kde=True,
                clip=(0, 1),
                color=current_col,
                alpha=0.3,
                fill=True,
                common_norm=False,
            )

        # Plot settings
        plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
        # plt.title(
        #    f"Predictive Distribution of a Single Patient\nLabel: {y_true}", y=1.25
        # )
        plt.xlabel("Predicted Risk")
        # plt.ylabel("Probability")
        # #plt.grid()
        plt.gca().spines["right"].set_color("none")
        plt.gca().spines["top"].set_color("none")
        plt.savefig(
            os.path.join(save_path, f"use_case_i_predictions_{title}.pdf"),
            dpi=300.0,
            bbox_inches="tight",
        )


def sorted_predictions_with_threshold(
    predictive_distribution: np.ndarray,
    save_path: str,
    t: float = 0.5,
    std_factor: float = 1.0,
    use_quantile: bool = False,
    quantile: float = 0.95,
) -> None:
    """Creates a plot of sorted predictions with the uncertainty intervals around the prediction
    Args:
        predictive_distribution (np.ndarray): predictive distribution of a single model
        save_path (str): path where the plot is saved
        t (float, 0.5): example threshold
        std_factor (float, 1.0): multiplier of the standard deviation to quantify uncertainty
        use_quantiles (bool, true): use quantile ranges of the inverse CDF, rather than standard deviation
        quantile (float, 0.95): quantile if `use_quantile`is set to true
    Returns None
    """
    figure(figsize=(8, 8))
    mean = predictive_distribution.mean(0)
    sort = np.argsort(mean)
    mean = mean[sort]

    if use_quantile:
        quantified_uncertainty_string = f"{int(quantile*100)}%-credible interval"
        high = np.quantile(
            predictive_distribution, quantile, axis=0, method="inverted_cdf"
        )
        low = np.quantile(
            predictive_distribution, 1 - quantile, axis=0, method="inverted_cdf"
        )
        high = high[sort]
        low = low[sort]
        std = np.vstack([mean - low, high - mean])

    else:
        quantified_uncertainty_string = (
            str(int(std_factor)) + "$\sigma$" if std_factor != 1 else "$\sigma$"
        )
        std = std_factor * predictive_distribution.std(0)
        std = std[sort]
        high = mean + std
        low = mean - std

    mask = ~((high > t) & (low < t))

    plt.plot(
        range(len(mean)),
        t * np.ones_like(mean),
        marker=".",
        label=f"Classification Threshold (t={t}, coverage={mask.mean():.2f})",
    )
    mean_string = "$\\bar{y}$"
    plt.errorbar(
        np.arange(len(mean))[~mask],
        mean[~mask],
        yerr=std[:, ~mask] if use_quantile else std[~mask],
        marker=".",
        alpha=0.2,
        label=f"Risk score ({mean_string}) and uncertainty ({quantified_uncertainty_string}): uncertain classification",
        ls="none",
    )
    plt.errorbar(
        np.arange(len(mean))[mask],
        mean[mask],
        yerr=std[:, mask] if use_quantile else std[mask],
        marker=".",
        alpha=0.2,
        label=f"Risk score ({mean_string}) and uncertainty ({quantified_uncertainty_string}): certain classification",
        ls="none",
    )

    # plt.title(
    #    f"Sorted Predictions with Uncertainties for the Horseshoe-MH Model\nCoverage: {mask.mean():.2f}"
    # )
    plt.ylabel("Risk Probability")
    plt.xlabel("Index")
    plt.legend()
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    # plt.grid()
    plt.savefig(
        os.path.join(save_path, f"sorted_plot_{use_quantile}.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_model_coverages(
    preds: Dict,
    thresholds: List,
    y_true: np.ndarray,
    save_path: str,
    std_factor: float = 1.0,
    use_quantile: bool = False,
    quantile: float = 0.95,
) -> None:
    """Creates plots of the coverage vs f1-score, recall, and precision across models
    Args:
        preds (Dict): dictionary of predictive distributions with model name as key and their distributions of the test set as value
        thresholds (List): list of thresholds to iterate over
        y_true (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
        std_factor (float, 1.0): multiplier of the standard deviation to quantify uncertainty
        use_quantiles (bool, true): use quantile ranges of the inverse CDF, rather than standard deviation
        quantile (float, 0.95): quantile if `use_quantile`is set to true
    Returns:
        None
    """
    lasso_preds = preds.pop("Frequentist LASSO")
    preds["Frequentist LASSO"] = np.array([lasso_preds])

    metrics = [f1_score, precision_score, recall_score]
    metric_names = ["F1 Score", "Precision (PPV)", "Recall (Sensitivity)"]
    titles = ["a", "b", "c"]

    for i, (metric, metric_name, title) in enumerate(
        zip(metrics, metric_names, titles)
    ):
        figure(figsize=(5, 5))
        thresholds.append(np.round(y_true.mean(), 2))
        for model, pred in preds.items():
            y_pred = pred.mean(0)
            if use_quantile:
                high = np.quantile(pred, quantile, axis=0, method="inverted_cdf")
                low = np.quantile(pred, 1 - quantile, axis=0, method="inverted_cdf")
            else:
                std = std_factor * pred.std(0)
                high = y_pred + std
                low = y_pred - std
            coverages = []
            ppvs = []
            for t in sorted(thresholds):
                mask = ~((high > t) & (low < t))
                ppv = metric(y_true[mask], y_pred[mask] > t)
                coverage = mask.mean()
                coverages.append(coverage)
                ppvs.append(ppv)
            plt.scatter(ppvs, coverages, marker="x", label=f"{model}", s=60)
            plt.plot(ppvs, coverages, linestyle="--")
            for i, txt in enumerate(sorted(thresholds)):
                plt.annotate(f"t={txt}", (ppvs[i], coverages[i]))
        del thresholds[-1]
        # plt.title(f"{metric_name} vs Coverage")
        plt.legend()
        plt.xlabel(f"{metric_name}")
        plt.ylabel("Coverage")
        plt.gca().spines["right"].set_color("none")
        plt.gca().spines["top"].set_color("none")
        # plt.grid()
        plt.savefig(
            os.path.join(save_path, f"coverage_vs_metric_{title}_{use_quantile}.pdf"),
            dpi=300.0,
            bbox_inches="tight",
        )


def plot_uncertainty_coverages(
    predictive_distribution: np.ndarray,
    thresholds: List,
    y_true: np.ndarray,
    save_path: str,
    std_factors: List = [1.0, 2.0],
    quantiles: List = [0.95, 0.99],
) -> None:
    """Creates plots of the coverage vs f1-score, recall, and precision across various uncertainty ranges
    Args:
        predictive_distribution (np.ndarray): single predictive distribution of a Bayesian model
        thresholds (List): list of thresholds to iterate over
        y_true (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
        std_factors (List): list of factors to multiple the standard deviation with
        quantiles (List): upper quantiles for the inverse CDF calculation
    Returns:
        None
    """
    metrics = [f1_score, precision_score, recall_score]
    metric_names = ["F1 Score", "Precision (PPV)", "Recall (Sensitivity)"]
    color_palette = ["#A85CF9", "#5534A5", "#4B7BE5", "#6FDFDF"]
    titles = ["a", "b", "c"]

    for i, (title, metric, metric_name) in enumerate(
        zip(titles, metrics, metric_names)
    ):
        figure(figsize=(5, 5))
        cols = iter(color_palette)
        thresholds.append(np.round(y_true.mean(), 2))
        y_pred = predictive_distribution.mean(0)
        for std in std_factors:
            current_cols = next(cols)
            y_uncerts = predictive_distribution.std(0) * std
            coverages = []
            ppvs = []
            for t in sorted(thresholds):
                mask = ~((y_pred + y_uncerts > t) & (y_pred - y_uncerts < t))
                ppv = metric(y_true[mask], y_pred[mask] > t)
                coverage = mask.mean()
                coverages.append(coverage)
                ppvs.append(ppv)
            plt.scatter(
                ppvs,
                coverages,
                marker="x",
                label=f"{int(std) if std!=1 else ''}$\sigma$",
                s=60,
                color=current_cols,
            )
            plt.plot(ppvs, coverages, linestyle="--", color=current_cols)
            for i, txt in enumerate(sorted(thresholds)):
                plt.annotate(f"t={txt}", (ppvs[i], coverages[i]))
        for quant in quantiles:
            current_cols = next(cols)
            high = np.quantile(
                predictive_distribution, quant, axis=0, method="inverted_cdf"
            )
            low = np.quantile(
                predictive_distribution, 1 - quant, axis=0, method="inverted_cdf"
            )
            coverages = []
            ppvs = []
            for t in sorted(thresholds):
                mask = ~((high > t) & (low < t))
                ppv = metric(y_true[mask], y_pred[mask] > t)
                coverage = mask.mean()
                coverages.append(coverage)
                ppvs.append(ppv)
            plt.scatter(
                ppvs,
                coverages,
                marker="x",
                label=f"{int(quant*100)}% - credible interval",
                s=60,
                color=current_cols,
            )
            plt.plot(ppvs, coverages, linestyle="--", color=current_cols)
            for i, txt in enumerate(sorted(thresholds)):
                plt.annotate(f"t={txt}", (ppvs[i], coverages[i]))
        del thresholds[-1]
        # plt.title(f"{metric_name} vs Coverage Horseshoe MH")
        plt.legend()
        plt.xlabel(f"{metric_name}")
        plt.ylabel("% Data Classified")
        plt.gca().spines["right"].set_color("none")
        plt.gca().spines["top"].set_color("none")
        # plt.grid()
        plt.savefig(
            os.path.join(save_path, f"coverage_vs_metric_scale_quantiles_{title}.pdf"),
            dpi=300.0,
            bbox_inches="tight",
        )


def plot_certain_posteriors(
    posterior: np.ndarray,
    frequentist_lasso: LogisticRegression,
    column_names: List,
    save_path: str,
    model_name: str,
    quantile: float = 0.95,
) -> None:
    """Plots the distributions of the certain posteriors, with a credible interval of `quantile` over or below 0.
    Args:
        posterior (np.ndarray): posterior [n_samples, n_features]
        frequentist_lasso (sklearn.linear_models.LogisticRegression): frequentist model to find out frequentist coefficient counterpart
        column_names (List): list of names of the corresponding features
        save_path (str): path where the figure will be stored
        model_name(str): name of the model
        quantiles (float: 0.95): quantiles for credible intervals
    """

    feature_dict = {}
    frequentist_features = []
    for feat, freq_feat, name in zip(
        posterior["beta"].T, frequentist_lasso.coef_[0], column_names
    ):
        low = np.quantile(feat, 1 - quantile, method="inverted_cdf")  # mean - std
        high = np.quantile(feat, quantile, method="inverted_cdf")  # mean + std
        if not (high > 0 and low < 0):  # and "PROC_" not in name:
            new_name = " ".join(
                [
                    x.capitalize()
                    if name.split("_")[0] not in ["LABS", "PROC", "DX"]
                    else f"{name.split('_')[0].upper()}: {name.split('_')[1]}"
                    for x in name.split("_")[1:]
                ]
            )
            feature_dict[new_name] = feat
            frequentist_features.append(freq_feat)

    if not feature_dict:
        return

    axes = az.plot_forest(
        feature_dict,
        kind="ridgeplot",
        combined=True,
        ridgeplot_truncate=False,
        ridgeplot_quantiles=[0.5],
        ridgeplot_overlap=3,
        colors="#00A1D5FF",
        figsize=(6, 10),
    )
    axes[0].scatter(
        list(reversed(frequentist_features)),
        4.04 * np.arange(len(frequentist_features)),
        marker="x",
        label="Frequentist LASSO\ncoefficients",
        color="#B24745FF",
        s=80,
    )
    plt.grid()
    plt.legend()
    plt.axvline(x=0)
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    # plt.title(
    #    f"Posterior Distribution of {int(quantile*100)}% - Credible Variables\n{model_name} model"
    # )
    file_name = "".join(model_name.split(" ")).lower()

    plt.savefig(
        os.path.join(save_path, f"variables_posterior_{file_name}.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_group_uncertainty(
    predictive_distribution: np.ndarray,
    feature_matrix: pd.DataFrame,
    names_dict: Dict,
    group_name: str,
    save_path: str,
    std_factor: float = 1.0,
    use_quantile: bool = False,
    quantile: float = 0.95,
    rotation: int = 0,
    order: bool = False,
) -> None:
    """Plot senstivity analysis for group uncertainty. Creates box plots with 0.25, 0.5, and 0.75 quantile, as well as outliers, according to groups
    NOTE: this works only for group > 2. If you wish to compare a single one-hot encoded group, please refer to `plot_group_uncertainty_binary`.
    Args:
        predictive_distribution (np.ndarray): predictive distribution of a model
        feature_matrix (pd.DataFrame): dataframe containing the feautures (unnormalized) with the corresponding columns
        names_dict (Dict): dictionary containing the feature matrix column names as keys and the formatted clean names as values
        group_name (str): name of the group that is being inspected
        save_path (str): path where the figure will be stored
        std_factor (float, 1.0): multiplier of the standard deviation to quantify uncertainty
        use_quantiles (bool, true): use quantile ranges of the inverse CDF, rather than standard deviation
        quantile (float, 0.95): quantile if `use_quantile`is set to true
        rotation (int, 0): roation of the names labels
        order (bool, False): keeps order of the boxes
    Returns:
        None
    """
    figure(figsize=(10, 4))
    data_df = feature_matrix[names_dict.keys()]
    if use_quantile:
        quantified_uncertainty_string = f"{int(quantile*100)}%-credible interval"
    else:
        quantified_uncertainty_string = (
            str(int(std_factor)) + "$\sigma$" if std_factor != 1 else "$\sigma$"
        )

    # Calculate the quantified uncertainty
    uncertainty_col_name = f"Uncertainty ({quantified_uncertainty_string})"
    data_df[uncertainty_col_name] = (
        predictive_distribution.std(0) * std_factor
        if not use_quantile
        else (
            np.quantile(
                predictive_distribution,
                1 - ((1 - quantile) / 2),
                axis=0,
                method="inverted_cdf",
            )
            - np.quantile(
                predictive_distribution,
                (1 - quantile) / 2,
                axis=0,
                method="inverted_cdf",
            )
        )
    )
    data_df.rename(
        columns={x: f"{y} (n={data_df[x].sum()})" for x, y in names_dict.items()},
        inplace=True,
    )

    # reverse one-hot encoding
    data_df[group_name] = (data_df.iloc[:, : len(names_dict)] == 1).idxmax(1)
    sns.boxplot(
        x=group_name,
        y=uncertainty_col_name,
        data=data_df,
        order=data_df.columns[: len(names_dict)] if order else None,
    )

    # Calculate Kruskal-Wallis statistic
    groups = [
        data_df[data_df[group_name] == name][uncertainty_col_name]
        for name in data_df.columns[: len(names_dict)]
    ]
    stat, p = stats.kruskal(*groups)
    p = np.round(p, 3) if p > 0.001 else "< 0.001"
    # plt.title(
    #    f"{uncertainty_col_name} Distribution by {group_name} for Horseshoe-MH\n Kruskal-Wallis Test p-value: {p}"
    # )
    # plt.grid()
    plt.xticks(rotation=rotation)
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.savefig(
        os.path.join(
            save_path, f"uncertainty_{'_'.join(group_name.lower().split())}.pdf"
        ),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_group_uncertainty_binary(
    predictive_distribution: np.ndarray,
    feature_matrix: pd.DataFrame,
    col_group_name: str,
    group_name: str,
    binary_dict: Dict,
    save_path: str,
    std_factor: float = 1.0,
    use_quantile: bool = False,
    quantile: float = 0.95,
) -> None:
    """Plot senstivity analysis for group uncertainty. Creates box plots with 0.25, 0.5, and 0.75 quantile, as well as outliers, according to groups
    NOTE: this works only for group == 2. If you wish to compare multiple one-hot encoded group, please refer to `plot_group_uncertainty`.
    Args:
        predictive_distribution (np.ndarray): predictive distribution of a model
        feature_matrix (pd.DataFrame): dataframe containing the feautures (unnormalized) with the corresponding columns
        col_group_name (str): column name of the group to be inspected
        group_name (str): name of the group that is being inspected
        binary_dict (Dict): Dictionary mapping `0` to the proper label and `1` to the proper label for one hot encoding
        save_path (str): path where the figure will be stored
        std_factor (float, 1.0): multiplier of the standard deviation to quantify uncertainty
        use_quantiles (bool, true): use quantile ranges of the inverse CDF, rather than standard deviation
        quantile (float, 0.95): quantile if `use_quantile`is set to true
    Returns:
        None
    """
    figure(figsize=(10, 4))
    data_df = feature_matrix[[col_group_name]]
    yes = int(data_df[col_group_name].sum())
    no = int(len(data_df[col_group_name]) - data_df[col_group_name].sum())
    data_df[col_group_name] = data_df[col_group_name].map(
        {1: f"{binary_dict[1]} (n={yes})", 0: f"{binary_dict[0]} (n={no})"}
    )

    # Create title for uncertainty axis
    if use_quantile:
        quantified_uncertainty_string = f"{int(quantile*100)}%-credible interval"
    else:
        quantified_uncertainty_string = (
            str(int(std_factor)) + "$\sigma$" if std_factor != 1 else "$\sigma$"
        )

    # Calculate the quantified uncertainty
    uncertainty_col_name = f"Uncertainty ({quantified_uncertainty_string})"
    data_df[uncertainty_col_name] = (
        predictive_distribution.std(0) * std_factor
        if not use_quantile
        else (
            np.quantile(
                predictive_distribution,
                1 - ((1 - quantile) / 2),
                axis=0,
                method="inverted_cdf",
            )
            - np.quantile(
                predictive_distribution,
                (1 - quantile) / 2,
                axis=0,
                method="inverted_cdf",
            )
        )
    )
    data_df.rename(columns={col_group_name: group_name}, inplace=True)

    # Plot the boxplot
    sns.boxplot(x=group_name, y=uncertainty_col_name, data=data_df)

    # Calculate Kruskal-Wallis statistic
    groups = [
        data_df[data_df[group_name] == i][uncertainty_col_name]
        for i in [f"{binary_dict[1]} (n={yes})", f"{binary_dict[0]} (n={no})"]
    ]
    stat, p = stats.kruskal(*groups)
    p = np.round(p, 3) if p > 0.001 else "< 0.001"
    # plt.title(
    #    f"{uncertainty_col_name} Distribution by Depression for Horseshoe-MH\n Kruskal-Wallis Test p-value: {p}"
    # )
    # plt.grid()
    plt.gca().spines["right"].set_color("none")
    plt.gca().spines["top"].set_color("none")
    plt.savefig(
        os.path.join(
            save_path, f"uncertainty_{'_'.join(group_name.lower().split())}.pdf"
        ),
        dpi=300.0,
        bbox_inches="tight",
    )
