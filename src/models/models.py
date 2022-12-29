""" This file contains the Bayesian Logistic regression used to replicate the Logistic LASSO
Author:
    Claudio Fanconi
"""
import numpy as np
import pymc3 as pm
import theano


class Laplace_Prior(pm.Model):
    def __init__(
        self, X_i: theano.shared, y_i: theano.shared, d: int, name="", model=None
    ) -> None:
        """Initialize the Bayesian logistic LASSO with a Laplacian prior
        Args:
            X_i (theano shared ): variable for the feature matrix
            y_i (theano shared ): variable for the labels
            d (int): dimensions of the features
            name (str, ""): name of the model
            model (None): model instance
        """
        super().__init__(name, model)
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        # define the model using reparametrization
        beta = pm.Laplace("beta", mu=0, b=1 / np.sqrt(2), shape=d)
        b = pm.Laplace("b", mu=0, b=1 / np.sqrt(2))
        p = pm.Deterministic("p", pm.math.invlogit(X_i.dot(beta) + b))
        pm.Bernoulli("y", p=p, observed=y_i)


class Horseshoe_Prior(pm.Model):
    def __init__(
        self, X_i: theano.shared, y_i: theano.shared, d: int, name="", model=None
    ) -> None:
        """Initialize the Bayesian logistic LASSO with a parametrized Horseshoe+ prior
        Args:
            X_i (theano shared ): variable for the feature matrix
            y_i (theano shared ): variable for the labels
            d (int): dimensions of the features
            name (str, ""): name of the model
            model (None): model instance
        """
        super().__init__(name, model)
        # call super's init first, passing model and name
        # to it name will be prefix for all variables here if
        # no name specified for model there will be no prefix
        # define the model using reparametrization
        ν = 1
        r_local = pm.Normal("r_local", mu=0, sd=1.0, shape=d)
        rho_local = pm.InverseGamma(
            "rho_local", alpha=0.5 * ν, beta=0.5 * ν, shape=d, testval=0.1
        )
        r_global = pm.Normal("r_global", mu=0, sd=1)
        rho_global = pm.InverseGamma("rho_global", alpha=0.5, beta=0.5, testval=0.1)
        tau = r_global * pm.math.sqrt(rho_global)
        lambda_ = r_local * pm.math.sqrt(rho_local)
        z = pm.Normal("z", mu=0, sd=1, shape=d)
        beta = pm.Deterministic("beta", z * lambda_ * tau)
        b = pm.Normal("b", mu=0, sd=1)
        p = pm.Deterministic("p", pm.math.invlogit(X_i.dot(beta) + b))
        pm.Bernoulli("obs", p=p, observed=y_i)
