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

# Regression

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as T
import theano
from copy import copy 

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams['figure.figsize']=(12,8)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
degree = 180.0/np.pi
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## A Motivating Example: Linear Regression

From [Getting started with PyMC3](https://docs.pymc.io/notebooks/getting_started.html)

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$y_i \sim N(\vec{\beta} \cdot \vec{x_i}+\alpha,\sigma)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha_t, sigma_t = 1, 1
beta_t = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha_t + beta_t[0]*X1 + beta_t[1]*X2 + np.random.randn(size)*sigma_t
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
map_estimate = pm.find_MAP(model=basic_model)

map_estimate
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with basic_model:
    trace = pm.sample(tune=1000, draws=20000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with basic_model:
    az.plot_trace(trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
az.summary(trace).round(2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
betas=trace.posterior['beta'].as_numpy()
```

```{code-cell} ipython3
plt.hexbin(betas[0,:,0], betas[0,:,1]);
plt.scatter(beta_t[:1], beta_t[1:],color='red',s=50);
plt.colorbar();
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Data analysis recipes: Fitting Model to data, David W. Hong.

+++ {"tags": []}

From [Data analysis recipes: Fitting Model to data, David W. Hong](https://arxiv.org/abs/1008.4686)

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
data = np.loadtxt("linear_regression.txt")
clean_data = data[5:]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o');
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$$y_i \sim N(\vec{\beta} \cdot \vec{x_i}+\alpha,\sigma_i),\quad \sigma_i \text{ known}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
y_model = pm.Model()

with y_model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
   
    # Expected value of outcome
    mu = alpha + beta*clean_data[:,0]

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=clean_data[:,2], observed=clean_data[:,1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
map_estimate = pm.find_MAP(model=y_model)
map_estimate
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ys = map_estimate['alpha']+map_estimate['beta']*xs
plt.plot(xs,ys);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with y_model:
    # draw 500 posterior samples
    trace = pm.sample(tune=1000, draws=10000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with y_model:
    az.plot_trace(trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
az.summary(trace).round(2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
trace.stack(sample=['chain', 'draw'], inplace=True)

post=trace.posterior
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
post;
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots()
ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ys = map_estimate['alpha']+map_estimate['beta']*xs
plt.plot(xs,ys,'orange');
ys = np.mean(post['alpha'].data)+np.mean(post['beta'].data)*xs
plt.plot(xs,ys,'red');
ys = post['alpha'].median().data+ post['beta'].median().data*xs
plt.plot(xs,ys,'green');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
for i in range(64):
    k  =np.random.randint(0, len(trace))
    ys = post['alpha'][k].data+post['beta'][k].data*xs
    plt.plot(xs,ys,'grey', alpha=0.25);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
plt.hexbin(post['alpha'].data, post['beta'].data);
plt.xlabel('$\\alpha$', fontsize=20);
plt.ylabel('$\\beta$', fontsize=20);
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Uncertainties on both axes

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def covariance(arr):
    cov = np.zeros((2,2))
    cov[0,0]=arr[1]*arr[1]
    cov[1,1]=arr[0]*arr[0]
    cov[1,0]=cov[0,1] = arr[0]*arr[1]*arr[2]
    return cov
    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def ellipse_par(cov):
    e,v = np.linalg.eig(cov)
    c = v[0,0]
    angle = np.arccos(c)
    return (np.sqrt(e[0]), np.sqrt(e[1]),angle)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
covs = np.apply_along_axis(covariance,1, data[:,2:])
clean_covs=covs[5:]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
sigmas = np.linalg.inv(covs)
clean_sigmas = sigmas[5:]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def ellipse_patch( x, y, w, h, a):
    return Ellipse(xy=(x,y), width=2*w, height=2*h, angle=a*degree)    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots() 
ax.set_xlim(0,300)
ax.set_ylim(0,700)
ax.scatter(clean_data[:,0], clean_data[0:,1],marker='.')
for d in clean_data[:]:
    c = covariance(d[2:])
    ep = ellipse_par(c)
    epa =  ellipse_patch(*d[0:2], *ep)
    epa.set_facecolor('none')
    epa.set_edgecolor('r')
    ax.add_patch(epa)
plt.close()    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def o_d(phi,s):
    cos  = np.cos(phi)
    sin  = np.sin(phi)
    return  (np.array([s*cos, s*sin]), 
                      np.array([-sin, cos])
                     )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
n = len(clean_data)
with pm.Model() as model:
    s   = pm.HalfFlat('s')
    phi = pm.Uniform('phi',lower = -np.pi, upper = np.pi)  
    t = pm.Flat('t',shape=n)
    o = T.stack([s*np.cos(phi), s*np.sin(phi)])
    d = T.stack([-np.sin(phi), np.cos(phi)])
   
    p = pm.Deterministic('p', d*t.reshape((-1,1))+o)
    
    for i in range( n ):
        obs = pm.MvNormal('obs_%i' % (i,) , mu = p[i] , cov = clean_covs[i], 
                      observed = clean_data[i,0:2])
   
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
with model:
    trace = pm.sample(draws=10000, tune=10000, chains=4, return_inferencedata=True)
```

```{code-cell} ipython3
trace.stack(sample=['chain', 'draw'], inplace=True)
post2 = trace.posterior
p_m = post2['p'].data.mean(2)
```

```{code-cell} ipython3
with model:
    az.plot_trace(trace, var_names=["phi","s"]);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
phi_m = post2["phi"].data.mean()
s_m = post2["s"].data.mean()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
fig, ax = plt.subplots() 
ax.set_xlim(0,300)
ax.set_ylim(0,700)
ax.scatter(clean_data[:,0], clean_data[0:,1],marker='.')
for d in clean_data[:]:
    c = covariance(d[2:])
    ep = ellipse_par(c)
    epa =  ellipse_patch(*d[0:2], *ep)
    epa.set_facecolor('none')
    epa.set_edgecolor('r')
    ax.add_patch(epa)
o,d  = o_d(phi_m,s_m)  
times = np.linspace(-600,-100,100)
ps = o+d*times.reshape(-1,1)
ax.plot(ps[:,0], ps[:,1]);
ax.scatter(p_m[:,0], p_m[:,1],color='green');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---

```
