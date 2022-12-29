""" This file contains the testing script for creating the various plots for the experiments with quantified uncertainty
Author:
    Claudio Fanconi
"""
import os
import pandas as pd
import numpy as np
from src.models.models import Horseshoe_Prior
from src.utils.config import config
from src.utils.data_preprocessing import preprocessing
from src.utils.metrics import print_results
from src.utils.plots import (
    plot_certain_posteriors,
    plot_model_coverages,
    plot_uncertainty_coverages,
    single_predictions,
    sorted_predictions_with_threshold,
    uncertainty_vs_predictions,
)
import pymc3 as pm
import theano


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
    # Load test and train data
    X_train, _, y_train, y_test = preprocessing(
        feature_path=config.data.data_path,
        label_path=config.data.label_path,
        train_ids_path=config.data.train_ids,
        test_ids_path=config.data.test_ids,
        outcome=config.data.label_type,
    )

    # load the predictions / predictive distributions of the four models:
    frequentist_LASSO = np.load(
        os.path.join(config.data.save_predictions, "frequentist_LASSO_predictions.npz"),
        allow_pickle=True,
    )["arr_0"]
    laplace_vi = np.load(
        os.path.join(
            config.data.save_predictions, "laplace_vi_predictive_distribution.npz"
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
    predictive_distributions = {
        "Laplace-VI": laplace_vi,
        "Laplace-MH": laplace_mh,
        "Horseshoe-MH": horseshoe_mh,
        "Frequentist LASSO": frequentist_LASSO,
    }

    # Create plots for 3 single patient predictions
    single_predictions(
        predictive_distributions.copy(), y_test, config.data.figures_path
    )

    # Create scatter plot for uncerainty vs predicted risk
    uncertainty_vs_predictions(
        predictive_distributions.copy(), config.data.figures_path
    )

    # Create sorted predictions with uncertainties around them to visualize the coverage
    sorted_predictions_with_threshold(
        horseshoe_mh, config.data.figures_path, t=config.uncertainty.arbitrary_threshold
    )

    # Create coverage vs classification plot over models
    plot_model_coverages(
        predictive_distributions.copy(),
        config.uncertainty.thresholds,
        y_test,
        config.data.figures_path,
    )

    # Create coverage vs classification plot over uncertainty thresholds for single predictive distribution
    plot_uncertainty_coverages(
        horseshoe_mh, config.uncertainty.thresholds, y_test, config.data.figures_path
    )

    # Create posterior plot:
    # Set Theano shared variables
    X_i = theano.shared(X_train)
    y_i = theano.shared(y_train.values)
    d = X_train.shape[1]

    with pm.Model() as BLL:
        # Define model in context
        Horseshoe_Prior(X_i, y_i, d)
        horseshoe_mh_posterior = pm.load_trace(
            os.path.join(config.data.save_posteriors, "horseshoe_mh"), model=BLL
        )

    plot_certain_posteriors(
        horseshoe_mh_posterior, feature_matrix.columns, config.data.figures_path
    )


if __name__ == "__main__":
    main(random_state=config.seed)
