""" This file contains the various metric calculations, especially designated for calculating 95%-confidence intervals with bootstrapping
Author:
    Claudio Fanconi
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import matplotlib.pyplot as plt
import calibration as cal
from typing import Tuple, Dict


def bootstrap(df, func=roc_auc_score) -> Tuple:
    """Bootstrap for calculating the confidence interval of a metric function
    Args:
        df (pd.DataFrame): dataframe containing 'predictions' and ' outcomes'
        func (function): metric function that takes (y_true, y_pred) as parameters
    Returns:
        lower, upper 95% confidence interval
        full bootstrap
    """
    aucs = []
    for i in range(1000):
        sample = df.sample(
            n=df.shape[0], random_state=i, replace=True
        )  # take 80% for the bootstrap
        aucs.append(func(sample["outcomes"], sample["predictions"]))

    return np.percentile(np.array(aucs), 2.5), np.percentile(np.array(aucs), 97.5), aucs


def bayesian_ci(y_true: np.ndarray, y_preds: np.ndarray, func=roc_auc_score):
    """Create CI directly from the Bayesian sampled predictions, rather than bootstrapping it
    Args:
        y_true (np.ndarray): true labels, shape [#test]
        y_preds (np.ndarray): input predictions, shape [sample draws, #test]
        func (Callable): metric function that takes (y_true, y_pred) as parameters
    Returns:
        lower, upper 95% confidence interval
    """
    aucs = []
    for y_pred in y_preds:
        aucs.append(func(y_true, y_pred))

    return np.percentile(np.array(aucs), 2.5), np.percentile(np.array(aucs), 97.5)


def ece(y_true, y_proba):
    """computes the calibration error
    Args:
        y_true(pd.Dataframe): true labels
        y_pred(np.array): predicted probabilities
    Returns:
        the calibration error of the model
    """
    return cal.get_calibration_error(y_proba, y_true.values.astype(int))


def prediction_entropy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """computes the entropy of the predictions.
    Args:
        y_true (np.ndarray): ground thruths
        y_proba (np.ndarray): predicted probabilities
    """
    return -np.mean(y_proba * np.log(y_proba) + (1 - y_proba) * np.log(1 - y_proba))


def treat_all_curve(
    ax,
    y_true: np.ndarray,
    n_thresholds: int = 100,
    lower_t: float = 0.0,
    upper_t: float = 1.0,
    title: str = "",
    lowest_net_benefit: float = -0.5,
) -> None:
    """Creates a net benefit curve over various thresholds when treating all the patients (baseline)
    Args:
        y_true (np.ndarray): true risk score
        n_thresholds (int, 100): number of thresholds that should be checked
        lower_t (float, 0.0): lower threshold boundary
        upper_t (float, 1.0): upper threshold boundary
    Returns:
        Net benefit curve plotted
        None
    """
    ts = np.linspace(lower_t, upper_t, n_thresholds)
    net_benefits = []
    event_rate = y_true.mean()

    for t in ts:
        net_benefit = event_rate - (1 - event_rate) * t / (1 - t)
        if net_benefit >= lowest_net_benefit:
            net_benefits.append(net_benefit)
        else:
            net_benefits.append(np.nan)
    ax.plot(ts, np.array(net_benefits), label=title)


def treat_none_curve(
    ax,
    n_thresholds: int = 100,
    lower_t: float = 0.0,
    upper_t: float = 1.0,
    title: str = "",
) -> None:
    """Creates a net benefit curve over various thresholds when treating none the patients (baseline)
    Args:
        y_true (np.ndarray): true risk score
        n_thresholds (int, 100): number of thresholds that should be checked
        lower_t (float, 0.0): lower threshold boundary
        upper_t (float, 1.0): upper threshold boundary
    Returns:
        Net benefit curve plotted
        None
    """
    ts = np.linspace(lower_t, upper_t, n_thresholds)
    ax.plot(ts, np.zeros_like(ts), label=title)


def net_benefit_curve(
    ax,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 100,
    lower_t: float = 0.0,
    upper_t: float = 1.0,
    title: str = "",
) -> None:
    """Creates a net benefit curve over various thresholds
    Args:
        y_true (np.ndarray): true labels
        y_score (np.ndarray): predicted risk scores
        n_thresholds (int, 100): number of thresholds that should be checked
        lower_t (float, 0.0): lower threshold boundary
        upper_t (float, 1.0): upper threshold boundary
    Returns:
        Net benefit curve plotted
        None
    """
    ts = np.linspace(lower_t, upper_t, n_thresholds)
    net_benefits = []
    for t in ts:
        net_benefits.append(net_benefit(y_true, y_score, t))
    ax.plot(ts, np.array(net_benefits), label=title)


def net_benefit(
    y_true: np.ndarray, y_score: np.ndarray, threshold: np.float64 = 0.5
) -> np.float64:
    """Calculates the net benefit score for a given threshold
    Args:
        y_true (np.ndarray): true labels
        y_score (np.ndarray): predicted risk scores
        threshold (np.float): current threshold
    Returns:
        net_benefit_score (np.float): calculated net benefit with given threshold
    """
    N = len(y_true)
    TP = (y_true == (y_score >= threshold)) & (y_true == True)
    FP = (y_true != (y_score >= threshold)) & ((y_score >= threshold) == True)
    return (TP.sum() - threshold / (1 - threshold) * FP.sum()) / N


def print_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Print all the performance metrics (AUROC, AUPRC, LL, ECE, Entropy)
    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): predicted risk scores
    Returns:
        Dict of the bootstraps
    """
    results = {"predictions": y_pred, "outcomes": y_true}
    y_pred = y_pred
    results_df = pd.DataFrame(data=results)
    bootstraps = []
    low_95, high_95, bootstrap_auroc = bootstrap(results_df)
    print(
        f"AUROC: {roc_auc_score(y_true, y_pred):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
    )
    low_95, high_95, bootstrap_auprc = bootstrap(
        results_df, func=average_precision_score
    )
    print(
        f"AUPRC: {average_precision_score(y_true, y_pred):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
    )
    low_95, high_9, bootstrap_ce = bootstrap(results_df, func=log_loss)
    print(
        f"Cross-Entropy: {log_loss(y_true, y_pred):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
    )
    low_95, high_95, bootstrap_ece = bootstrap(results_df, func=ece)
    print(f"ECE: {ece(y_true, y_pred):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})")
    print("--------------------------")
    return {
        "AUROC": bootstrap_auroc,
        "AUPRC": bootstrap_auprc,
        "Log-Loss": bootstrap_ce,
        "ECE": bootstrap_ece,
    }