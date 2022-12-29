# ACU Uncertainty Estimation
***
## Name
Predictive Uncertainty with Bayesian Logistic Regression to Identify Chemotherapy Patients at Risk for Acute Care

## Description
This is the implementation for Bayesian logistic regression to predict the risk of acute care use (ACU) and the quantification of predictive uncertainty.

## Abstract
Logistic regression, especially with $\ell_1$-penalty (also known as LASSO), is widely used in medical informatics to predict the risk of an event for a patient (e.g. hospitalisation, mortality, use of acute care). However, in the frequentist inference approach, it is often not possible to quantify the uncertainty of the predictions. In this paper, we explore Bayesian logistic LASSO (BLL) models to identify patients at risk for acute care use after starting chemotherapy, based on their high-dimensional electronic health records. The added advantage of these Bayesian models over the frequentist model is that each prediction is represented as a distribution that we can use to estimate predictive uncertainty, rather than relying on a point estimate. In our experiments, we show how uncertainty can be used when a risk model is used in practice, e.g. for automatic treatment classification. Our results show that BLL with adequate priors and a posterior approximated with Metropolis-Hastings sampling is a suitable alternative to classical logistic LASSO. It achieves the same prediction results and is also able to quantify the prediction uncertainty. Furthermore, we show that patient subgroups can be biased in the quantified uncertainty.

## Cite Us

```
@article{fanconi2022uncertainty,
    title={Predictive Uncertainty with Bayesian Logistic Regression to Identify Chemotherapy Patients at Risk for Acute Care}, 
    author={Claudio Fanconi and Anne de Hond and Dylan Peterson and Angelo Capodici and Tina Hernandez-Boussard},
    year={2022},
    booktitle={...},
}
```

## Installation
Clone the current repository
```
git clone https://code.stanford.edu/fanconic/acu-uncertainty-estimation
cd acu-uncertainty-estimation
```

First you need to install PyMC3. For this, we suggest following the installation steps on their official [webpage](https://pypi.org/project/pymc3/).

We suggest to create a virtual environment and install the required packages.
```
conda create -n pymc_env -c conda-forge python=3.10.4 libpython mkl-service numba python-graphviz scipy arviz
conda activate pymc_env
conda install -r requirements.txt
```

### Source Code Directory Tree
```
.
├── data                # Folder with synthetic data
├── figures             # Figures of the experiments
├── saved_posteriors    # Saved posterior traces for the experiments
└── src                 # Source code            
    ├── models              # Bayesian logistic regression models
    └── utils               # Useful functions, such as loggers and config

```


## Running the Experiments
To run the models, you first need to prepare the data. For this experiment we expect four CSV files: `feature_matrix.csv` shall contain the features, `labels.csv` should contain the labels. Both of these should be indexed by a patient deidentifier number. `test_ids.csv` and `train_ids.csv` are CSV files that contain the patiend deid files of the test and training set, respectively. You can change the paths in `config.yml` file. In this file you can also set which model should be fitted, by setting their flags to either True or False

To fit the models, and create predictions of the test set, run 
```
python ./fit_models.py
```

To run the experiments and compare the models on their predictive peformance (metrics, calibration, net benefit), run
```
python ./test_model_predictions.py
```

To run the experiments and compare the models uncertainties, run
```
python ./test_model_uncertainties.py
```

To test the sensitivity of predictive uncertainty, run
```
python ./test_sensitivity.py
```
## Authors
- Claudio Fanconi (fanconic@ethz.ch)
- Anne de Hond
- Dylan Peterson
- Angelo Capodici
- Tina Hernandez-Boussard
