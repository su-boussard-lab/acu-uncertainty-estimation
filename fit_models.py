""" This file contains the training script for the Bayesian logistic regression models
Author:
    Claudio Fanconi
"""
import os
import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence

import theano
from sklearn.linear_model import LogisticRegression

from src.utils.config import config
from src.utils.data_preprocessing import preprocessing
from src.models.models import Horseshoe_Prior, Laplace_Prior


def main(random_state: int = 42) -> None:
    """Main function which trains the deep learning model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """

    # Load test and train data
    X_train, X_test, y_train, _ = preprocessing(
        feature_path=config.data.data_path,
        label_path=config.data.label_path,
        train_ids_path=config.data.train_ids,
        test_ids_path=config.data.test_ids,
        outcome=config.data.label_type,
    )

    # Fit frequentist LASSO:
    if config.model.frequentist_LASSO:
        print("Fitting frequentist LASSO")
        frequentist_LASSO = LogisticRegression(
            penalty="l1", max_iter=1000, solver="liblinear", C=0.02
        )
        frequentist_LASSO.fit(X_train, y_train)

        print("Predicting frequentist LASSO")
        frequentist_LASSO_predictions = frequentist_LASSO.predict_proba(X_test)[:, 1]

        # Save predictions
        np.savez(
            os.path.join(
                config.data.save_predictions, "frequentist_LASSO_predictions.npz"
            ),
            frequentist_LASSO_predictions,
        )

    # ------------ Laplace prior BLR with Variational Inference ------------------------------
    if config.model.laplace_vi:
        # Set Theano shared variables
        X_i = theano.shared(X_train)
        y_i = theano.shared(y_train.values)
        d = X_train.shape[1]

        print("Fitting Laplace prior BLR with Variational Inference")
        with pm.Model() as BLL:

            # Define model in context
            Laplace_Prior(X_i, y_i, d)

            if config.model.pretrained:
                laplace_vi_posterior = pm.load_trace(
                    os.path.join(config.data.save_posteriors, "laplace_vi"), model=BLL
                )
            else:
                # Perform ADVI
                callback = CheckParametersConvergence(diff="absolute")
                steps = 3000
                posterior = pm.fit(
                    n=steps,
                    callbacks=[callback],
                    random_seed=random_state,
                    method="fullrank_advi",
                )
                laplace_vi_posterior = posterior.sample(1000)

                # Save posterior
                pm.save_trace(
                    trace=laplace_vi_posterior,
                    directory=os.path.join(config.data.save_posteriors, "laplace_vi"),
                    overwrite=True,
                )

        print("Predicting Laplace prior BLR with Variational Inference")
        X_i.set_value(X_test)
        laplace_vi_predictive_distribution = pm.sample_posterior_predictive(
            laplace_vi_posterior, 10000, BLL, var_names=["p"], random_seed=11
        )

        # Save predictive distrubtions
        np.savez(
            os.path.join(
                config.data.save_predictions, "laplace_vi_predictive_distribution.npz"
            ),
            laplace_vi_predictive_distribution["p"],
        )

    # ------------ Laplace prior BLR with Metrpolis-Hastings ------------------------------
    if config.model.laplace_mh:
        # Set Theano shared variables
        X_i = theano.shared(X_train)
        y_i = theano.shared(y_train.values)
        d = X_train.shape[1]

        print("Fitting Laplace prior BLR with Metrpolis-Hastings")
        with pm.Model() as BLL:

            # Define model in context
            Laplace_Prior(X_i, y_i, d)

            if config.model.pretrained:
                laplace_mh_posterior = pm.load_trace(
                    os.path.join(config.data.save_posteriors, "laplace_mh"), model=BLL
                )
            else:
                # Perform MH-sampling
                warmup = 2000
                num_samples = 2000
                step = pm.Metropolis()
                laplace_mh_posterior = pm.sample(
                    num_samples, step=step, tune=warmup, random_seed=random_state
                )

                # Save posterior
                pm.save_trace(
                    trace=laplace_mh_posterior,
                    directory=os.path.join(config.data.save_posteriors, "laplace_mh"),
                    overwrite=True,
                )

        print("Predicting Laplace prior BLR with Metrpolis-Hastings")
        X_i.set_value(X_test)
        laplace_mh_predictive_distribution = pm.sample_posterior_predictive(
            laplace_mh_posterior, 10000, BLL, var_names=["p"], random_seed=11
        )

        # Save predictive distrubtion
        np.savez(
            os.path.join(
                config.data.save_predictions, "laplace_mh_predictive_distribution.npz"
            ),
            laplace_mh_predictive_distribution["p"],
        )

    # ------------   Horseshoe+ prior BLR with Metropolis-Hastings ------------------------------
    if config.model.horseshoe_mh:
        # Set Theano shared variables
        X_i = theano.shared(X_train)
        y_i = theano.shared(y_train.values)
        d = X_train.shape[1]

        print("Fitting Horseshoe+ prior BLR with Metropolis-Hastings")
        with pm.Model() as BLL:

            # Define model in context
            Horseshoe_Prior(X_i, y_i, d)

            if config.model.pretrained:
                horseshoe_mh_posterior = pm.load_trace(
                    os.path.join(config.data.save_posteriors, "horseshoe_mh"), model=BLL
                )
            else:
                # Perform MH-sampling
                warmup = 2000
                num_samples = 2000
                step = pm.Metropolis()
                horseshoe_mh_posterior = pm.sample(
                    num_samples, step=step, tune=warmup, random_seed=random_state
                )

                # Save posterior
                pm.save_trace(
                    trace=horseshoe_mh_posterior,
                    directory=os.path.join(config.data.save_posteriors, "horseshoe_mh"),
                    overwrite=True,
                )

        print("Predicting Horseshoe+ prior BLR with Metropolis-Hastings")
        X_i.set_value(X_test)
        horseshoe_mh_predictive_distribution = pm.sample_posterior_predictive(
            horseshoe_mh_posterior, 10000, BLL, var_names=["p"], random_seed=11
        )

        # Save predictive distrubtion
        np.savez(
            os.path.join(
                config.data.save_predictions, "horseshoe_mh_predictive_distribution.npz"
            ),
            horseshoe_mh_predictive_distribution["p"],
        )


if __name__ == "__main__":
    main(random_state=config.seed)
