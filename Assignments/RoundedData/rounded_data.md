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
import numpy as np
```

# Rounded data  (from "Bayesian Data Analysis")

+++

It is a common problem for measurements to be observed in rounded form. For a simple example, suppose we weigh an object five times and measure weights, rounded to the nearest pound, of 10, 10, 12, 11, 9. Assume the unrounded measurements are normally distributed with a noninformative prior distribution on the mean $\mu$ and variance $\sigma^2$.

+++

$$\mu\propto 1\qquad \sigma^2\propto \frac{1}{\sigma^2}$$

+++

## Problem 1

+++

Give the posterior distribution for $(\mu, \sigma^2)$ obtained by pretending that the observations are exact unrounded measurements.

+++

## Problem 2

+++

Give the correct posterior distribution for $(\mu, \sigma^2)$ treating the measurements as rounded.

+++

## Problem 3

+++

How do the incorrect and correct posterior distributions differ? Compare means, variances, and contour plots.

+++

### Contour plots

+++

To make a contour plot of a function of two variables we need its values dsitributed over a grid. This function takes a function and x and y ranges (arrays of values) and returns a grid with x values corresponding to columns and y values to rows

```{code-cell} ipython3
def make_grid(f,xs,ys):
    grid = np.zeros((len(ys), len(xs)))
    for iy in range(len(ys)):
         for ix in range(len(xs)):
                grid[iy,ix]=f(xs[ix],ys[iy])
            
    return grid    
```

```{code-cell} ipython3
def quad (x,y):
    return -x*x-0.25*y*y 
```

```{code-cell} ipython3
xs = np.linspace(-3,3,100)
ys = np.linspace(-3,3,100)
grid = make_grid(quad, xs,ys)
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
plt.contour(xs,ys,grid)
plt.show()
```

We can specify contour levels

```{code-cell} ipython3
plt.contour(xs,ys,grid, levels=[-4,-3,-2,-1,0])
plt.show()
```

and fill areas between them

```{code-cell} ipython3
plt.contourf(xs,ys,grid, levels=[-4,-3,-2,-1,0])
plt.show()
```

When solving the problem is better to calculate not the posterior probability but the log of posterior probability.

+++

#### Marginal distribution

+++

To calculate mean and variance of $\mu$ and $\sigma^2$ we need marginal distributions. We can approximate them  numerically by summing over one axis of the grid.
Do not forget to exponentiate the log of probability before summing!
