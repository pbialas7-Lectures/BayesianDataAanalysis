---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Bicycle trafic

```{code-cell} ipython3
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = [12,8]

import pymc3 as pm
import arviz as az
```

Analysis of proportions: a survey was done of bicycle and other vehicular traffic in the neighborhood of the campus of the University of California, Berkeley, in the spring of 1993. Sixty city blocks were selected at random; each block was observed for one hour, and the numbers of bicycles and other vehicles traveling along that block were recorded. The sampling was stratified into six types of city blocks: busy, fairly busy, and residential streets, with and without bike routes, with ~ten blocks measured in each stratum. The  displays the number of bicycles and other vehicles recorded in the study. For this problem only the data on residential streets is diplayed.

```{code-cell} ipython3
bikelane = np.array([[16,58],[9,90],[13,57],[19,103],[20,57],[18,86],[17,112],[35,273],[55,64]])
no_bikelane  = np.array([[12,113],[1,18],[2,14],[4,44],[9,208],[7,67],[9,29],[8,154]])
```

In each array the first colum contains the number of bikes and the second colum the number of other vehicles.

+++

We will be also working with proportions:

```{code-cell} ipython3
y = bikelane[:,0]/(bikelane[:,0]+bikelane[:,1])
z = no_bikelane[:,0]/(no_bikelane[:,0]+no_bikelane[:,1])
```

## Problem 1

+++

Assume that in each array data is distributed according to binomial distribution with same $p$ (bicycle probability) for all blocks. Please write down the posterior probability for $p$. Assume the uniform non-informative prior. What distribution it is?

+++

$$p(p|b_i, o_i)\propto\prod_i \binom{b_i+o_i}{b_i} p^{b_i}(1-p)^{o_i}=\prod_i \binom{b_i+o_i}{b_i} p^{\sum_i b_i}(1-p)^{\sum_i o_i}$$

+++

Draw 1000 samples of $p$ from this distribution. Then for each $p$ draw 8 (bikelane) or 9 (no bikelane) binomial samples with same numbers of vehicles as observed.  E.g. for bikelane for each $p$ draw samples with n = [16+58, 9+90, ...].

+++

Compare this sample with actual data. Does  the model give an adequate description of data?

+++

## Problem 2

+++

Let $y_i$ and $z_i$ be the observed proportion of traffic that was on bicycles in the residential streets with bike lanes and with no bike lanes, respectively (so $y_1$ = 16/(16 + 58) and $z_1$ = 12/(12 + 113), for example). Set up a model so that the $y_i$’s are independent and identically distributed according to  $Beta(\alpha_y, \beta_y)$ distribution and the $z_i$’s are independent and identically distributed according to $Beta(\alpha_z, \beta_z)$ distribution.

```{code-cell} ipython3
plt.plot(y,'o',label='bike lane')
plt.plot(z,'o', label='no bike lane');
plt.legend();
```

Assume prior distribution in $\alpha$ and $\beta$ as follows:

+++

Let $\mu=\frac{\alpha}{\alpha+\beta}$ and $\kappa=\alpha+\beta$. Then assume that
$\mu$ is distributed variable on $[0,1]$ and  $\kappa$ is exponentialy  distributed $P(\kappa) \propto \exp(-\lambda \kappa)$. Choose $\lambda$ as tu ensure sufficiently wide distribution for $\kappa$.

+++

Set up the corresponing model in PyMC3. Simulate the posterior predictive samples (use `pm.sample_posterior_predictive` function) and check the model. Is the model a good fit to the data?

+++

## Problem 4

+++

Let $\mu_y = E(\tilde{y}_i|\alpha_y, \beta_y)$ be the mean of the predictive distribution of the $\tilde{y}_i$’s; Similarly, define $\mu_z$. Using your posterior simulations plot a histogram of the posterior simulations of $\mu_y - \mu_z$, the expected difference in proportions in bicycle traffic on residential streets with and without bike lanes.
