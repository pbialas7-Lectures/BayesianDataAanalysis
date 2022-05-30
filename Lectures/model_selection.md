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

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%load_ext autoreload
%autoreload 2 
```

+++ {"slideshow": {"slide_type": "slide"}}

# Model selection

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy  as np
import scipy as sp
import scipy.stats as st
from scipy.special import gamma

import pymc3 as pm
import arviz as az

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Students distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
xs = np.linspace(-5,5,1000)
fig, ax = plt.subplots()
for nu in (1,2,5,10,25,50):
    ax.plot(xs, st.t.pdf(xs, df=nu), label=f"$\\nu={nu}$");
ax.plot(xs, st.norm.pdf(xs), color = 'black', label="$\mathcal{{N}}(0,1)$")    
ax.legend();    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
mu = 1 
xs = np.linspace(-5,5,1000)
fig, ax = plt.subplots()
for s in (0.2,0.5, 1,2,5):
    ax.plot(xs, st.t.pdf(xs, df=1, loc=mu, scale=s), label=f"$s={s}$");
ax.legend();    
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## True model

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
true_nu = 4
true_mu = 1
true_scale = 2
true_dist = st.t(loc=1, scale=2, df=4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
true_mean, true_std = true_dist.mean(), true_dist.std()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
print(true_mean, true_std, np.sqrt(true_nu/(true_nu-2))*2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
sample = true_dist.rvs(size=10000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
plt.hist(sample, bins = 100, density=True, histtype='step');
ts = np.linspace(-15,15,200)
plt.plot(ts, true_dist.pdf(ts), color='red', label='true')
plt.plot(ts, st.norm(true_mu, scale=true_scale).pdf(ts), color='gray', label=f"$\mathcal{{N}}(1,{st.norm(true_mu, scale=true_scale).std()})$")
plt.plot(ts, st.norm(true_mu, true_std).pdf(ts), color='gray', label=f"$\mathcal{{N}}(1,{true_std:.2f})$")
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Data

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
y_size = 59
y = true_dist.rvs(size=y_size, random_state=14543)
y_mean = y.mean()
y_s = y.std(ddof=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
plt.plot(y,'.');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
st.t.fit(y)
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Normal model

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
with pm.Model() as normal_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.Normal('y_obs', mu = mu, sd = sigma, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    MAP = pm.find_MAP()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
MAP
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    normal_trace = pm.sample(draws=4000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    az.plot_trace(normal_trace);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Posterior Predictive Check

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Posterior predictive distribution

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

* Sample $\mu$ and $\sigma$ from posterior
* Sample $\hat y$ from $\mathcal{N}(\mu,\sigma)$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$p(\hat y| y) = \int p(\hat y|\theta) p_{post}(\theta|y)\text{d}\theta$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$p(\hat y) = \int \underbrace{p(\hat y|\mu,\sigma)}_{\mathcal{N}(\mu, \sigma)} p_{post}(\mu,\sigma)\text{d}\mu\text{d}\sigma$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
normal_trace.posterior['mu']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
normal_trace.posterior['mu'].shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
posterior_sample = np.stack( (normal_trace.posterior['mu'],normal_trace.posterior['sigma']),-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
posterior_sample.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
posterior_predictive_sample = np.apply_along_axis(lambda a: st.norm.rvs(*a, size=59),2,posterior_sample)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
posterior_predictive_sample.reshape(16000,59)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
pps = pm.sample_posterior_predictive(trace=normal_trace, model=normal_model)
```

```{code-cell} ipython3

```

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

```python 
with pm.Model() as normal_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.Normal('y_obs', mu = mu, sd = sigma, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
pps['y_obs']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(np.min(pps['y_obs'],axis=1) , histtype='step', density=True, bins=50);
plt.hist(np.min(posterior_predictive_sample,axis=2).ravel() , histtype='step', density=True, bins=50);
plt.axvline(y.min(), color='black');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(np.max(pps['y_obs'],axis=1) , histtype='step', density=True, bins=50);
plt.hist(np.max(posterior_predictive_sample,axis=2).ravel() , histtype='step', density=True, bins=50);
plt.axvline(y.max(), color='black');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(st.kurtosis(pps['y_obs'],bias=False, axis=1) , histtype='step', density=True, bins=50);
plt.axvline(st.kurtosis(y, bias=False), color='black');
```

+++ {"slideshow": {"slide_type": "slide"}}

## Out of sample predictive fit

+++ {"slideshow": {"slide_type": "-"}}

$$\log p_{post}(\tilde{y}|y)  = \log \int p(\tilde{y}|\theta)p_{post}(\theta|y)\text{d}\theta,\qquad \tilde y \sim f $$

+++ {"slideshow": {"slide_type": "slide"}}

### Expected log predictive density

+++

$$E_f(\log p_{post}(\tilde{y})|y) = \int \log p_{post}(\tilde{y}|y) f(\tilde{y})\text{d}\tilde{y}_i $$

+++ {"slideshow": {"slide_type": "slide"}}

### Expected log pointiwise predictive density

+++

$$\sum_i E_f(\log p_{post}(\tilde{y}_i)|y) = \sum_i \int \log p_{post}(\tilde{y}_i|y) f(\tilde{y_i})\text{d}\tilde{y}_i $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
theta_posterior_sample = np.stack((normal_trace.posterior['mu'].to_numpy().ravel(), normal_trace.posterior['sigma'].to_numpy().ravel()) , axis=1)
```

+++ {"slideshow": {"slide_type": "slide"}}

### log pointwise predictive density (lppd)

+++

$$\sum_{i=1}^n \log \int p(\tilde y_i|\theta)p(\theta|y)\text{d}\theta$$

+++ {"tags": []}

### computed lppd

+++

$$\sum_{i=1}^n \log\left(\frac{1}{N}\sum_{s=1}^N p(\tilde y_i|\theta^s)\right),\qquad \theta \sim p_{post}(\theta|y)$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

### Expected predictive log distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def normal_likelihood(mu, sigma,y):
        return st.norm(loc=mu, scale=sigma).pdf(y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def post(y, theta_sample):
    return np.apply_along_axis(lambda theta: normal_likelihood(*theta,y).mean(), 1, theta_sample).mean(axis=0) 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
y_size * np.log(post(true_dist.rvs(size=1000), theta_sample=theta_posterior_sample[:1000])).mean()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
st.norm.pdf(true_dist.rvs(1), loc= theta_posterior_sample[:,0], scale=theta_posterior_sample[:,1]).mean()
```

+++ {"slideshow": {"slide_type": "skip"}, "tags": []}

### Computed log pointwise predictive distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
np.log(st.norm(theta_posterior_sample[:,0].reshape(-1,1), theta_posterior_sample[:,1].reshape(-1,1)).pdf(y).mean(0)).sum()
```

+++ {"slideshow": {"slide_type": "slide"}}

##  Leave-one-out cross validation

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$\sum_i\log p(y_i|y_{-i})$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
with normal_model:
    loo=pm.loo(normal_trace,  pointwise=False)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
loo
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Widely  applicable information criterion

+++ {"slideshow": {"slide_type": "fragment"}}

$$\widehat{\text{elppd}}_{WAIC}= \text{lppd}-p_{WAIC}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$-2(\text{lppd}-p_{WAIC})$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$\text{computed }p_{WAIC} =\sum_{i=1}^n Var_{s}[\log p(y_i|\theta^s)]$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    waic=az.waic(normal_trace, pointwise=False)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
waic
```

+++ {"slideshow": {"slide_type": "slide"}}

## Slightly better  model

```{code-cell} ipython3
with pm.Model() as student_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.StudentT('y_obs', mu = mu, sd = sigma, nu=4, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with student_model:
    MAP = pm.find_MAP()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
MAP
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with student_model:
    student_trace = pm.sample(draws=4000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with student_model:
    az.plot_trace(student_trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
pps = pm.sample_posterior_predictive(trace=student_trace, model=student_model, samples =2000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(np.min(pps['y_obs'],axis=1) , histtype='step', density=True, bins=50);
plt.axvline(y.min());
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(np.max(pps['y_obs'],axis=1) , histtype='step', density=True, bins=50);
plt.axvline(y.max());
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with student_model:
    waic = az.waic(student_trace)
waic    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.compare({'normal': normal_trace, 'student':student_trace})
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.compare({'normal': normal_trace, 'student':student_trace}, ic='waic')
```
