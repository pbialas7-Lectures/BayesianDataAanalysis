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

# Bicycles - Hierachical model

```{code-cell} ipython3
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,8]
import pymc3 as pm
import arviz as az
```

```{code-cell} ipython3
bikelane = np.array([[16,58],[9,90],[13,57],[19,103],[20,57],[18,86],[17,112],[35,273],[55,64]])
no_bikelane  = np.array([[12,113],[1,18],[2,14],[4,44],[9,208],[7,67],[9,29],[8,154]])
```

## Problem 1

+++

Set up a model for the data in  so that, for $j=1, \ldots $, the observed number of bicycles at location $j$ is binomial with unknown probability $\theta_j$ and sample size equal to the total number of vehicles (bicycles included) in that block. The parameter $\theta_j$ can be interpreted as the underlying or ‘true’ proportion of traffic at location $j$ that is bicycles.  Assign a beta population distribution for the parameters $\theta_j$. For hyperparameters use the same prior as in the non hierarchical case: Let $\mu=\frac{\alpha}{\alpha+\beta}$ and $\kappa=\alpha+\beta$. Then assume that
$\mu$ is uniformely distributed variable on $[0,1]$ and  $\kappa$ is exponentialy  distributed $P(\kappa) \propto \exp(-\lambda \kappa)$. Choose $\lambda$ as tu ensure sufficiently wide distribution for $\kappa$.  

+++

Set up the full PyMC model for the problem. Compare the model with the data by sampling from the predictive posterior distribution (use `pm.sample_posterior_predictive` function). 

+++

### Bikelanes

+++

### No bikelanes

+++

## Problem

+++

Compare the posterior distributions of the parameters $\theta_j$
to the raw proportions, (number of bicycles / total number of vehicles) in location j. E.g. draw w posterior sample $\tilde{\theta}_i$ and  for each $i$ caclculate mean and standard deviation of $\tilde{\theta}_i$.  How do the inferences from the posterior distribution differ from the raw proportions? Show this on a plot (use `plt.errorbar` function).

+++ {"tags": []}

## Problem

+++

Plot the histogram of $\bar\theta_y-\bar\theta_z$. Calculate the 95% HDR (Highest Density Region) for the difference $\bar{\theta}_y-\bar{\theta}_z$. How does it compare with non-hierachical  model?  Hint: You can use the `arviz.hdi` function.
