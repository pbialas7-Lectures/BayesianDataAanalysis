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
%load_ext autoreload
%autoreload 2
```

### Problem 2

+++

#### Estimation of two independent exeriments  (from "Bayesian Data Analysis")

+++

An experiment was performed on the effects of magnetic fields on the flow of calcium out of chicken brains. Two groups of chickens were involved: a control group of 32 chickens and an exposed group of 36 chickens. One measurement was taken on each chicken, and the purpose of the experiment was to measure the average flow $\mu_c$ in untreated (control) chickens and the average flow $\mu_t$ in treated chickens. The 32 measurements on the control group had a sample mean of 1.013 and a sample standard deviation of 0.24. The 36 measurements on the treatment group had a sample mean of 1.173 and a sample standard deviation of 0.20.

+++

#### Problem 2.1

+++

Assuming the control measurements were taken at random from a normal distribution with mean $\mu_c$ and variance $\sigma_c^2$, what is the posterior distribution of $\mu_c$? Similarly, use the treatment group measurements to determine the marginal posterior distribution of $\mu_t$. Assume an uniformative  prior distribution on $\mu_c$, $\mu_t, \sigma^2_c, \sigma^2_t \sim 1,1,\sigma_c^{-2}, \sigma_t^{-2}$.

+++

#### Problem 2.2

+++

What is the posterior distribution for the difference, $\mu_t-\mu_c$? To get this, you may sample from the independent $t$ distributions you obtained in part (1) above. Plot a histogram of your samples and give an approximate 95% posterior interval (highest density region) for $\mu_t-\mu_c$.

+++

### Random numbers

+++

To generate random numbers with given distribution we can use functions in the module <code>scipy.stats</code> ([docs](https://docs.scipy.org/doc/scipy/reference/stats.html)) or <code>numpy.random</code>.

```{code-cell} ipython3
import numpy as np
import scipy.stats as st
```

Here we generate 10000 random numbers from  normal distribution with mean 1 and standard deviation 2

```{code-cell} ipython3
randoms = st.norm(1,2).rvs(size=10000)
```

We will use <code>matplotlib</code> library for plotting

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

for example

```{code-cell} ipython3
plt.plot(randoms[:500],'.')
plt.show()
```

displays the "time series" of first 500 numbers.

+++

### Histograms

+++

More usefull way of visualising random number is the histogram

```{code-cell} ipython3
c,b,p = plt.hist(randoms,bins=50)
plt.show()
```

Apart from calculating and plotting the histogram this function returns the number of counts in each bin (variable `c` above), and list of bin edges (variable `b` above). Please note that `len(b)=len(c)+1` ([docs](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)).

+++

Often we will need count values normalized so that the integral (area) under the histogram is one, so it can be compared with probability distribution. This can be achived using the density argument:

```{code-cell} ipython3
c,b,p = plt.hist(randoms,bins=50, density=True, histtype='step')
xs = np.linspace(-7,7, 200)
plt.plot(xs, st.norm(1,2).pdf(xs))
plt.show()
```

### Highest density region

+++

The Highest Density Region (HDR) is defined as follows: given a density distribution $f(x)$ on sample sapce $X$, the $\alpha$ HDR is a subset $R(\alpha)$ of  $X$:

+++

$$R(f_\alpha)=\{x:f(x)\ge f_\alpha\}$$

+++

where $f_\alpha$ is the largest constant such that $P(x\in R(f_\alpha))\ge \alpha$.

+++

This was already explained in the notebook `highest_density_interval`. The associated function can be found in the file `Assignments/hdr.py`. You can import it, but first you have add its path to python path:

```{code-cell} ipython3
import sys
sys.path.append('..')
import hdr
```

Given the `c` and `b` obtained from the `plt.hist` function we can transform it to the representation used in `hdr`:

```{code-cell} ipython3
dist=c
xs = (b[1:]+b[:-1])/2 # centers of bins
```

```{code-cell} ipython3
hdr95 = hdr.hdr(xs, dist, 0.95) # return intervals, mass  and level p
```

```{code-cell} ipython3
hdr95
```

```{code-cell} ipython3
plt.hist(randoms,bins=50, density=True, histtype='step');
plt.fill_between(xs,dist,0,where = ( (xs>hdr95[0][0]) & (xs<=hdr95[0][1])), color='lightgray' );
```
