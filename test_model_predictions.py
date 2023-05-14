""" This file contains the testing script for the BLRs to calculate the predictive metrics, calibration and net benefit
Author:
    Claudio Fanconi
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict
from src.utils.config import config
from src.utils.metrics import print_results
from src.utils.plots import calibration_plot, net_benefit_plot, violin_plot


def transpose_dict(dct: Dict) -> Dict:
    """Transposes a dictionary of dictionaries
    Args: dct (Dict): a dict of dicts
    returns:
        dict of dicts with transposed keys
    """
    d = defaultdict(dict)
    for key1, inner in dct.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return d


def main(random_state: int = 42) -> None:
    """Main function which trains the deep learning model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """
    # Load relevant test data
    feature_matrix = feature_matrix = (
        pd.read_csv(config.data.data_path, low_memory=False)
        .sort_values(by="PAT_DEID")
        .set_index("PAT_DEID")
        .drop("DEMO_INDEX_PRE_CHE", axis=1)
    )
    outcomes = pd.read_csv(config.data.label_path).set_index("PAT_DEID")
    labels_all = outcomes[config.data.label_type].reindex(feature_matrix.index)
    test_ids = pd.read_csv(config.data.test_ids)["PAT_DEID"]

    X_test = feature_matrix.loc[test_ids]
    y_test = labels_all.loc[test_ids]

    # load the predictions / predictive distributions of the four models:
    frequentist_LASSO = np.load(
        os.path.join(config.data.save_predictions, "frequentist_LASSO_predictions.npz"),
        allow_pickle=True,
    )["arr_0"]
    frequentist_bootstrapped_LASSO = np.load(
        os.path.join(
            config.data.save_predictions,
            "frequentist_LASSO_bootstrapped_predictions.npz",
        ),
        allow_pickle=True,
    )["arr_0"]
    laplace_vi = np.load(
        os.path.join(
            config.data.save_predictions, "laplace_vi_predictive_distribution_2.npz"
        ),
        allow_pickle=True,
    )["arr_0"]
    laplace_mh = np.load(
        os.path.join(
            config.data.save_predictions, "laplace_mh_predictive_distribution.npz"
        ),
        allow_pickle=True,
    )["arr_0"]
    horseshoe_mh = np.load(
        os.path.join(
            config.data.save_predictions, "horseshoe_mh_predictive_distribution.npz"
        ),
        allow_pickle=True,
    )["arr_0"]

    # Print predictive performance of models:
    predictions = {
        "Laplace-VI": laplace_vi.mean(0),
        "Laplace-MH": laplace_mh.mean(0),
        "Horseshoe-MH": horseshoe_mh.mean(0),
        "Frequentist LASSO": frequentist_LASSO,
        # "Frequentist Bootstrapped LASSO": frequentist_bootstrapped_LASSO.mean(0),
    }

    # Calculate the predictive metrics
    bootstraps_metrics = {}
    for name, y_pred in predictions.items():
        print(name)
        bootstrap_dict = print_results(y_test, y_pred)
        bootstraps_metrics[name] = bootstrap_dict

    # Calculate Violin plot
    bootstraps_metrics = transpose_dict(bootstraps_metrics)
    for metric, bootstrap in bootstraps_metrics.items():
        violin_plot(metric, bootstrap, config.data.figures_path)

    # Create calibration plot
    calibration_plot(predictions, y_test, config.data.figures_path)

    # Create Net Benefit Curve
    net_benefit_plot(predictions, y_test, config.data.figures_path)


if __name__ == "__main__":
    main(random_state=config.seed)
