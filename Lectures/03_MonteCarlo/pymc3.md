---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

# Normal model and PyMC3

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
import numpy as np
from scipy.stats import norm
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Data

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
n = 100
mu_true = 1.0
sigma_true = 2.0

np.random.seed(1212331)
y = norm(mu_true, sigma_true).rvs(100)
y_bar = y.mean()
s2 = y.var(ddof=1)
print(y_bar, s2)
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## PyMC3 model

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
import pymc3 as pm
print(f"Running on PyMC3 v{pm.__version__}")
import arviz as az
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Known variance

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
normal_model1 = pm.Model()

with normal_model1:
    mu = pm.Flat('mu')   # Prior
    y_obs=pm.Normal('y_obs', mu=mu, sigma=sigma_true, observed=y) #likelihood
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Maximal a Posteriori

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
MAP1 =  pm.find_MAP(model=normal_model1)
print(MAP1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
with normal_model1:
    MAP1 =  pm.find_MAP(model=normal_model1)
print(MAP1)
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Sampling from posterior distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
trace1 = pm.sample(model=normal_model1,tune=1000, draws=3000, return_inferencedata=False)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
trace1['mu']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with normal_model1:
    az.plot_trace(trace1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with normal_model1:
    az.plot_posterior(trace1)
```

```{code-cell} ipython3
with normal_model1:
    summary1 = az.summary(trace1)
summary1    
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Unknown variance and flat prior on sigma

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
normal_model2 = pm.Model()

with normal_model2:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    y_obs=pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
pm.find_MAP(model=normal_model2)
```

```{code-cell} ipython3
y_bar
```

```{code-cell} ipython3
np.sqrt((n-1)/n*s2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
trace2=pm.sample(model=normal_model2,tune=1000, draws=3000, return_inferencedata=False)
```

```{code-cell} ipython3
with normal_model2:
    az.plot_trace(trace2)
```

```{code-cell} ipython3
with normal_model2:
    az.plot_posterior(trace2)
```

```{code-cell} ipython3
with normal_model2:
    print(az.summary(trace2))
```

## Unknown variance and  $\sigma^{-1}$ prior on $\sigma$. 

```{code-cell} ipython3
normal_model3 = pm.Model()

with normal_model3:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    sigma_prior = pm.Potential('sigma_prior', -np.log(sigma))
    
    y_obs=pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

```{code-cell} ipython3
pm.find_MAP(model=normal_model3)
```

```{code-cell} ipython3
print(np.sqrt((n-1)/(n+1)*s2) )
```

```{code-cell} ipython3
trace3=pm.sample(model=normal_model3,tune=1000, draws=3000, return_inferencedata=False)
```

```{code-cell} ipython3
with normal_model3:
    az.plot_trace(trace3)
```

```{code-cell} ipython3
with normal_model3:
    az.plot_posterior(trace3)
```

## Unknow variance and $\sigma^{-2}$ prior on $\sigma^{2}$

```{code-cell} ipython3
normal_model4 = pm.Model()

with normal_model4:
    mu = pm.Flat('mu')
    var = pm.HalfFlat('var')
    var_prior = pm.Potential('sigma_prior', -np.log(var))
    
    y_obs=pm.Normal('y_obs', mu=mu, sigma=np.sqrt(var), observed=y)
```

```{code-cell} ipython3
map4 = pm.find_MAP(model=normal_model4)
print(map4)
print(np.sqrt(map4['var']))
```

```{code-cell} ipython3
print(np.sqrt((n-1)/(n+2)*s2) )
```

```{code-cell} ipython3
trace4=pm.sample(model=normal_model4,tune=1000, draws=3000, return_inferencedata=False)
```

```{code-cell} ipython3
with normal_model4:
    az.plot_trace(trace4)
```

```{code-cell} ipython3
with normal_model4:
    az.plot_posterior(trace4)
```

```{code-cell} ipython3

```
