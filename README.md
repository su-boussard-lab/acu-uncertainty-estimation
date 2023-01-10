# ACU Uncertainty Estimation
***
## Name
A Bayesian Approach to Predictive Uncertainty in Chemotherapy Patients at Risk of Acute Care Use

## Description
This is the implementation for Bayesian logistic regression to predict the risk of acute care use (ACU) and the quantification of predictive uncertainty.

## Abstract
**Background**: Machine learning (ML) predictions are becoming increasingly integrated into medical practice. One commonly used method, $\ell_1$-penalised logistic regression (LASSO), can estimate patient risk for disease outcomes but is limited by only providing point estimates. Instead, Bayesian logistic LASSO regression (BLLR) models provide distributions for risk predictions, giving clinicians a better understanding of predictive uncertainty, but they are not commonly implemented. 

**Setting and Methods**: This study evaluates the predictive performance of different BLLRs compared to standard logistic LASSO regression, using real-world, high-dimensional, structured electronic health record (EHR) data from 8,439 cancer patients initiating chemotherapy at a comprehensive cancer centre. Multiple BLLR models were compared against a LASSO model using an 80-20 random split using 10-fold cross-validation to predict the risk of acute care utilisation (ACU) after starting chemotherapy. 

**Results**: The LASSO model predicted ACU with an area under the receiver operating characteristic curve (AUROC) of 0.806 (95\%-CI: 0.792 to 0.820). BLLR with a Horseshoe+ prior and a posterior approximated by Metropolis-Hastings sampling showed similar performance: 0.807 (95\%-CI: 0.793 to 0.821) and offers the advantage of uncertainty estimation for each prediction. In addition, BLLR could identify predictions too uncertain to be automatically classified. BLLR uncertainties were stratified by different patient subgroups, demonstrating that predictive uncertainties significantly differ across race, cancer type, and stage. 

**Conclusion**: BLLRs are a promising yet underutilised predictive tool that increases explainability by providing risk estimates while offering a similar level of performance to standard LASSO-based models. Additionally, these models can identify patient subgroups with higher uncertainty, which can augment clinical decision-making. Further validation of these models is needed in different populations and clinical use cases. Nonetheless, these findings demonstrate the feasibility of using BLLRs to predict clinical outcomes accurately.

## Cite Us

```
@article{fanconi2022uncertainty,
    title={A Bayesian Approach to Predictive Uncertainty in Chemotherapy Patients at Risk of Acute Care Use}, 
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
├── (data)                # Folder with data (default folder, to be created)
├── (figures)             # Figures of the experiments (default folder, to be created)
├── (saved_posteriors)    # Saved posterior traces for the experiments (default folder, to be created)
└── src                   # Source code            
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
