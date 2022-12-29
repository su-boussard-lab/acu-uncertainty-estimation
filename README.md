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
    author={Claudio Fanconi and Anne de Hond and Angelo Capodici and Tina Hernandez-Boussard},
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
- Angelo Capodici
- Tina Hernandez-Boussard


## Theory
### Bayesian Machine Learning
Bayesian Machine Learning is a systematic approach to constructing statistical models, based on Bayes’ Theorem on two random variables $A$ and $B$:
$
\begin{equation}
   P(B|A) = \frac{P(A|B)\cdot P(B)}{P(A)} 
\end{equation}
$
If we want to fit a binary risk classification model with a feature input matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ (where $n \in \mathbb{N}$ is the number of inputs, and $d \in \mathbb{N}$ is the number of features), a label vector $\mathbf{y} \in \{0,1\}^n$, and $\boldsymbol{\theta} \in \mathbb{R}^d$ is a $d$-dimensional vector of the model parameters, we rewrite the Bayes' Theorem in Equation 1, conditioned on our already existing inputs $\mathbf{X}$, as follows:
$
\begin{equation}
P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y}) = \frac{P(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta})\cdot P(\boldsymbol{\theta}|\mathbf{X})}{P(\mathbf{y}|\mathbf{X})} 
\end{equation}
$
$P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y})$ is the estimated **posterior** of the model parameters, based on the inputs and labels. $P(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta})$ is the **likelihood** of the labels, given the weights and input data. $P(\boldsymbol{\theta}|\mathbf{X})$ is the **prior** distribution on the model weights, which are assumed to be independent of the inputs, and therefore denoted as $P(\boldsymbol{\theta})$. Finally, $P(\mathbf{y}|\mathbf{X})$ is the **evidence**, which is a normalization constant of the posterior probability density.
The main goal in Bayesian ML is to estimate the posterior distribution, given the likelihood and the prior distribution. The most likely parameters of the posterior distribution yield than the so-called **Maximum A Posteriori** (MAP) estimate:
$
\begin{align}
\boldsymbol{\theta}^* &= \underset{\boldsymbol{\theta}}{\arg\max}\, P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y})\\
&= \underset{\boldsymbol{\theta}}{\arg\max}\, P(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta})\cdot P(\boldsymbol{\theta})
\end{align}
$
The evidence is dropped in the maximization equation, as it does not depend on the model parameters $\boldsymbol{\theta}$. To make a prediction of single data point $\mathbf{x} \in \mathbb{R}^d$, we can use the MAP and a link function $\hat{y} = f(\mathbf{x}, \boldsymbol{\theta}^*)$. In our case of binary risk classification, the link function corresponds to a sigmoid function, $f(\mathbf{x}, \boldsymbol{\theta}) = \frac{1}{1+\text{exp}(-\boldsymbol{\theta}^\top\mathbf{x})}$. Furthermore, the posterior can be used to receive a full predictive distribution, rather than just a point estimate, by marginalizing the model weights:
$
\begin{align}
P(\hat{y}|\mathbf{X}, \mathbf{y}, \mathbf{x}) = \int P(\hat{y}|\boldsymbol{\theta}, \mathbf{x}) \cdot P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y}) d\boldsymbol{\theta}
\end{align}
$
Unfortunately, in many cases, the posterior is not tractable, and cannot be calculated analytically, especially for high-dimensional feature space. In the subsections~\ref{variational_inf} and \ref{metropolis-hastings} we are going to discuss two methods that are computationally feasible to approximate the posterior distribution.

### Likelihood and Priors
Depending on the task at hand, it is necessary to choose the right distribution for the likelihood and prior. To achieve the Bayesian equivalent of a classical logistic regression for our case of binary risk classification, it is suited to assume that the labels are sampled from a Bernoulli distribution (coin flip distribution) for the likelihood term:
$
\begin{align}
y_i \sim P(y_i|\mathbf{X}, \boldsymbol{\theta}) &= Bernoulli(p_i)\\&=p_i^{y_i}\cdot(1-p_i)^{1-y_i}
\end{align}
$
In our case, the probability parameter $p_i$ is the risk probability of the $i$-th input ($i \in \{0,1,...,n\}$), denoted here as $\mathbf{X}_i$, that we can calculate with the sigmoid link function
$$p_i =  \frac{1}{1+\text{exp}(-\boldsymbol{\theta}^\top\mathbf{X}_i)}$$
On the other hand, the choice of the prior $P(\boldsymbol{\theta})$ is equivalent to choosing the regularization term of logistic regression.
An $\ell_1$-regularization penalty (also known as LASSO) is equivalent to finding the MAP by assuming a centered Laplace distribution on the $j$-th model parameter ($j \in \{0,1,...,d\}$) with a predefined scale parameter $b \in \mathbb{R}^+$
$
\begin{align}
    \theta_j \sim P(\theta_j) &= Laplace(0, b)\\ &= \frac{1}{2b}\text{exp}\bigg(-\frac{|\theta_j|}{b}\bigg)
\end{align}
$
This can be shown as follows:
$
\begin{align}
    \boldsymbol{\theta}^* &= \underset{\boldsymbol{\theta}}{\arg\max}\, P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y})\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \text{log}P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y})\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \text{log}\bigg(\frac{1}{Z}P(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta})\cdot P(\boldsymbol{\theta})\bigg)\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \text{log}P(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) + \text{log} P(\boldsymbol{\theta})\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \text{log}\bigg(\prod_{i=1}^nP(y_i|\mathbf{X}, \boldsymbol{\theta})\bigg) + \text{log}\bigg(\prod_{j=1}^dP(\theta_j)\bigg)\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \sum_{i=1}^n\text{log}P(y_i|\mathbf{X}, \boldsymbol{\theta}) + \sum_{j=1}^d\text{log}P(\theta_j)\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \sum_{i=1}^n\text{log}\bigg(p_i^{y_i}\cdot(1-p_i)^{1-y_i}\bigg) + \sum_{j=1}^d\text{log}\bigg(\frac{1}{2b}\text{exp}\bigg(-\frac{|\theta_j|}{b}\bigg)\bigg)\\
    &= \underset{\boldsymbol{\theta}}{\arg\max}\, \sum_{i=1}^ny_i\cdot\text{log}p_i + (1-y_i)\cdot\text{log}(1-p_i) + \sum_{j=1}^d\text{log}\frac{1}{2b} - \frac{1}{b}\sum_{j=1}^d|\theta_j|\\
    &= \underset{\boldsymbol{\theta}}{\arg\min}\, - \sum_{i=1}^ny_i\cdot\text{log}p_i + (1-y_i)\cdot\text{log}(1-p_i) + \frac{1}{b}\sum_{j=1}^d|\theta_j|
\end{align}
$
where the first sum is the binary cross entropy loss used to minimise logistic regression, and the second sum is the $\ell_1$-penalty on all the model parameters weighted by the regularization paramater $\lambda = \frac{1}{b}$. The factorisation of the join distributions of the labels and model parameters from Equation~9 to Equation~10, comes from the independence assumption of the labels and weights.

[1] introduced the Horseshoe+ prior, to inducing stronger sparsity in Bayesian generalized linear models. The Horseshoe+ prior is a hierarchical prior (using hyperpriors: priors on priors) of the following distributions:
$
\begin{align}
    \theta_j \sim P(\theta_j| \lambda_j, \tau) &= \mathcal{N}(0, \lambda_j^2\tau^2)\\
    \lambda_j &\sim t^+(0, 1)\\
    \tau &\sim t^+(0, 1)
\end{align}
$
where $t^+$ is a half-$t$ distribution (only the positive support of the $t$ distribution). In our experiments, we use a parametrization of the Horseshoe+, proposed by [2], as it is more robust for sampling than the one in Equation~\ref{horseshoe}, with the following hyperpriors and priors:\\
$
\begin{align}
    r_i^{\text{local}} &\sim \mathcal{N}(0,1)\\
    \rho_i^{\text{local}} &\sim \Gamma^{-1}(\frac{1}{2},\frac{1}{2})\\
    r^{\text{global}} &\sim \mathcal{N}(0,1)\\
    \rho^{\text{global}} &\sim \Gamma^{-1}(\frac{1}{2},\frac{1}{2})\\
    z &\sim \mathcal{N}(0,1)\\
    \lambda_i &= r_i^{\text{local}}\sqrt{\rho_i^{\text{local}}}\\
    \tau &= r^{\text{global}}\sqrt{\rho^{\text{global}}}\\
    \theta_i &= z\lambda_i\tau
\end{align}
$
where $\Gamma^{-1}$ is the inverse Gamma distribution.

### Variational Inference
Variational inference (VI) is a method that seeks to approximate an intractable distribution by a simple one, as close as possible [3, 4]. Here, we try to approximate the posterior with the variational family of Gaussian distributions, by finding the distribution that minimizes the Kullback-Leibler (KL) divergence with the posterior distribution. The KL-divergence is a measure of how one probability distribution is different from a second, reference probability distribution. Mathematically, we can say $\mathcal{Q} = \{q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}, \boldsymbol{\Sigma})\}$ is the family of Gaussian distributions with mean $\boldsymbol{\mu} \in \mathbb{R}^d$ and the covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{d\times d}$, and the optimization goal is:
$
\begin{align}
     q^*(\boldsymbol{\theta}) \in \underset{q \in \mathcal{Q}}{\arg\min}\,KL\bigg(q(\boldsymbol{\theta})||P(\boldsymbol{\theta}|\mathbf{X}, \mathbf{y})\bigg)
\end{align}
$
where $q^*$ is a Gaussian distribution, with a specific mean and covariance matrix, that minimizes the KL-divergence with posterior. As this is now a Gaussian, it is simple to directly sample from this approximated distribution. In our experiments we use Automatic Differentiation Variational Inference (ADVI) [5], to solve the optimization problem in Equation~\ref{advi_eq} via stochastic gradient descent.

### Metropolis-Hastings Sampling
Metropolis-Hastings (MH) is a Markov-Chain Monte Carlo (MCMC) method, that seeks to approximate an intractable distribution by obtaining a sequence of random samples from that distribution, by simulating a Markov Chain. The key idea is to create a sequence of sample values that are iteratively produced, with the distribution of the next sample being dependent only on the current sample value. If the Markov Chain is simulated sufficiently long, the resulting samples are drawn from a distribution that is very close to the intractable distribution, in our case the posterior. The Metropolis-Hastings algorithm picks at each iteration a candidate for the next sample and with some probability, the sample is either accepted (used as the next sample) or rejected (current sample is kept for the next iteration) [6, 7].

## References
- [1] Bhadra, A., Datta, J., Polson, N. G., and Willard, B. T. The horseshoe+ estimator of ultra-sparse signals. arXiv: Statistics Theory, 2015
- [2] Piironen, J. and Vehtari, A. Projection predictive variable selection using stan+r. arXiv: Methodology, 2015
- [3] Wainwright, M. J. and Jordan, M. I. Graphical models, exponential families, and variational inference. [electronic resource] / wainwright, martin j. 2008
- [4] Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. Journal of the American Statistical Association, 112:859 – 877, 2016
- [5] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., andBlei, D. M. Automatic differentiation variational inference. J. Mach. Learn. Res., 18:14:1–14:45, 2017
- [6] Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., and Teller, E. Equation of state calculations by fast computing machines. Journal of Chemical Physics, 21:1087–1092, 1953
- [7] Hastings, W. K. Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57 (1):97–109, 04 1970. ISSN 0006-3444. doi: 10.1093/biomet/57.1.97. URL https://doi.org/10.1093/biomet/57.1.97.