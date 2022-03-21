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

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}}

### Highest density region (HDR)

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
xs = np.linspace(-10,20,1000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
tags: []
---
from scipy.stats import norm
dist = norm(4,2).pdf(xs)
dist += norm(-2,1).pdf(xs)
dist/=  np.trapz(dist,xs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig,ax =plt.subplots()
ax.plot(xs, dist);
```

+++ {"slideshow": {"slide_type": "slide"}}

$\beta$ HDR is a region where at least $\beta$ of probability is concentrated and has smallest possible volume in the sample space, hence highest density. More formal definition given below.

+++ {"slideshow": {"slide_type": "-"}}

Let $P_X(p)$ be de density function of  some random variable $X$ with values in $R_X$. Let $R_X(p)$ be the subsets of $R_X$ such  that

+++ {"slideshow": {"slide_type": "-"}}

$$ R(p) = \{x\in R_X: P_X(x)\ge p\}$$

+++ {"slideshow": {"slide_type": "-"}}

The $\beta$ HDR is equal to $R(p_\beta)$ where $p_\beta$ is the largest constant such that

+++ {"slideshow": {"slide_type": "-"}}

$$P\left(x\in R(p_\beta)\right)\ge \beta$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig,ax =plt.subplots()
ax.plot(xs, dist);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
def R(p, dist):
    gt = dist>p
    if np.all(gt):
        return np.array((0,len(dist)))                        
    return np.where(np.logical_xor(gt[1:],gt[:-1]))[0]

def R_mass(Rp,xs, dist):
    assert (Rp.size % 2)==0
    intervals = Rp.reshape(-1,2)
    mass = 0.0
    for inter in intervals:
        mass += np.trapz(dist[inter[0]:inter[1]], xs[inter[0]:inter[1]])
    return mass    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def plot_rp(xs,dist,p):
    fig, ax = plt.subplots()
    ax.plot(xs,dist);
    ax.set_ylim(0,1.25*dist.max())
    Rp=R(p,dist)
    Rx = xs[Rp]
    mass = R_mass(Rp, xs, dist)
    ax.axhline(p)
    for itr in Rx.reshape(-1,2):
        ax.fill_between(xs,dist,0, where = ( (xs>itr[0]) & (xs<itr[1])),color='lightgray');
    ax.text(10,0.15,f"${mass:.2f}$", fontsize=14);    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
from ipywidgets import interactive, FloatSlider
f=lambda p: plot_rp(xs,dist,p)
interactive_plot = interactive(f, p=FloatSlider(min=0.0, max=0.25,step=1e-4, value=0.22, readout_format=".3f"))
output = interactive_plot.children[-1]
#output.layout.height = '650px'
interactive_plot
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
def hdri(xs, dist,beta, p_max=1, eps=1e-6):
    p_min =0.0
    while p_max-p_min>eps:
        p = (p_max+p_min)/2
        Rp = R(p,dist)
        mass = R_mass(Rp, xs, dist)
        if mass > beta:
            p_min = p;
        else:
            p_max = p
    return xs[R(p_max,dist)],R_mass(Rp,xs,dist)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig, ax = plt.subplots()
ax.plot(xs, dist);
Rx,mass=hdri(xs,dist,0.9)
for itr in Rx.reshape(-1,2):
    ax.fill_between(xs,dist,0, where = ( (xs>itr[0]) & (xs<itr[1])),color='lightgray');
ax.text(10,0.15,f"${mass:.2f}$", fontsize=14);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
from scipy.optimize import fmin
from scipy.stats import *

def HDIofICDF(dist_name, credMass=0.95, **args):
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass =  1.0 - credMass

    def intervalWidth(lowTailPr):
        return distri.ppf(credMass + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, credMass + HDIlowTailPr])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
from scipy.stats import beta
```

```{code-cell} ipython3
ps=np.linspace(0,1,1000)
fig,ax=plt.subplots()
ax.plot(ps, beta(20,5).pdf(ps));
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
hdr = HDIofICDF(beta,a=20,b=5)
ps=np.linspace(0,1,1000)
fig,ax=plt.subplots()
ax.plot(ps, beta(20,5).pdf(ps));
ax.fill_between(ps, beta(20,5).pdf(ps),0, where=((ps>hdr[0]) & (ps<hdr[1])), color='lightgray');
```
