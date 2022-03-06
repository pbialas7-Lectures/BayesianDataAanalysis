---
jupytext:
  cell_metadata_json: true
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
---
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats as st
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
dc='#1f77b4' #default color
```

+++ {"slideshow": {"slide_type": "slide"}}

# Bayesian Data Analysis

+++ {"tags": []}

> ### Statistical inference is concerned with drawing conclusions, from numerical data, about quantities that are not observed.
>  "Bayesian Data Analysis" A. Gelman, J. B. Carlin, H. S. Stern,  D. B. Dunson, A. Vehtari, D. B. Rubin

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

- "Bayesian Data Analysis" A. Gelman, J. B. Carlin, H. S. Stern,  D. B. Dunson, A. Vehtari, D. B. Rubin.
- "Data Analysis, a Bayesian Tutorial" D.S. Silva with J. Skiling.
- "Bayesian methods for Hackers, Probabilistic Programming adn Bayesian Inference" C. Davidson-Pilon  [[online](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)]
- "From Laplace to Supernova SN 1987A: Bayesian Inference in Astrophysics" T. J. Loredo. [[pdf](https://bayes.wustl.edu/gregory/articles.pdf)]

+++ {"tags": [], "slideshow": {"slide_type": "slide"}}

## Reverend Thomas Bayes' original example -- pool table

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
p =  0.31415926
y =  0.786
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def make_pool_figure(*, ylim=(0,1)):
    fig,ax = plt.subplots(figsize=(12,8))
    plt.subplots_adjust(bottom=0.2)
    ax.xaxis.set_visible(False);
    ax.yaxis.set_visible(False);
    plt.annotate('$p$', (p, -.1 ), fontsize=20, annotation_clip=False,ha='center');
    ax.set_xlim(0,1);ax.set_ylim(0,1); 
    pax=ax.twinx()
    pax.get_yaxis().set_visible(False)
    pax.set_ylim(*ylim)
    return fig, ax, pax
    
    

def plot_ball(ax, x,y,*,bc, bs=100, lc=None, draw_line=False, empty=False, **kwargs):
    fill=bc
    if empty:
        fill='white'
    ax.scatter(x,y,marker='o',s=bs, edgecolor=bc, color=fill, zorder=1, **kwargs);
    if draw_line:
        if lc:
            linecolor=lc
        else:
            linecolor=bc
        ax.axvline(x, color=linecolor, zorder=-1)
    return ax    

cs=np.asarray(['red','darkgreen'])

def plot_balls(ax,n,x,left,cs,*, draw_line=True, **kwargs):
    for i in range(n):
        plot_ball(ax, x[i,0], x[i,1],bc=cs[left[i]], bs=200, draw_line=draw_line, empty=True, **kwargs); 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig,ax,pax = make_pool_figure()
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Prior

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p)$$

+++ {"slideshow": {"slide_type": "slide"}}

### What is probability ?

+++ {"slideshow": {"slide_type": "slide"}}

>#### "One sees, from this Essay, that the theory of probabilities is basically just common sense reduced to calculus; it makes one appreciate with exactness that which accurate minds feel with a sort of instinct, often without being able to account for it."
> "Théorie Analytique des Probabilités" Pierre-Simon Laplace

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
xs = np.linspace(0,1,1000)
prior = np.vectorize(lambda x: 1.0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
pax.plot(xs,prior(xs), zorder=1, c='blue');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
np.random.seed(87)
x = st.uniform(loc=0, scale=1).rvs(size=(100,2))
left=(x[:,0]<=p) + 0
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
pax.plot(xs,prior(xs), zorder=1, c='blue');
plot_balls(ax, 1, x,left, cs)
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Sampling distribution/ likelihood

+++

$$P(r|p)=P(x>p)= 1-p\quad P(l|p)=P(x\le p)=p$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Conditional probability

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(A|B) = \frac{P(A\cap B)}{P(B)}$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

We are rolling two dices, what is the probability that a three apeared in the results?

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from itertools import product
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
O = list(product(range(1,7), range(1,7)) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
A = list(filter(lambda x: x[0]==3 or x[1]==3, O))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
print(A)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from fractions import Fraction
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
print(Fraction(len(A),len(O)) )
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

How does that probability changes when we know that the sum of the to results is odd? What if the result is even?

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
rem = 1
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
B = list(filter(lambda x: (x[0]+x[1])%2 == rem, O))
len(B)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
AcapB = list(filter(lambda x: (x[0]+x[1])%2 == rem, A))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
print(Fraction(len(AcapB),len(B)))
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Product rule

+++

$$P(A\cap B)= P(A|B)P(B)$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$$B = \bigcup_{i}  B_i,\; \bigvee_{i\neq_j} B_i\cap B_j=\emptyset \implies P(B)=\sum_i P(B_i)$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(A\cap B)= \sum_i P(A|B_i)P(B_i)$$

+++

### Height distribution

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(h)=P(h|f)P(f)+P(h|m)P(m)$$

+++ {"tags": [], "slideshow": {"slide_type": "slide"}}

## Bayes' theorem

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(B|A)= \frac{P(A\cap B)}{P(A)}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$\boxed{P(B|A)=\frac{P(A|B)P(B)}{P(A)}}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Prior & posterior

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(p|r)=\frac{P(r|p)P(p)}{P(r)}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(r)=\int_{0}^1\text{d}p P(r|p)P(p)= \int_0^1\text{d}p (1-p)= 1-\frac{1}{2}=\frac{1}{2}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(p|r)=2-2p$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
posteriors = [lambda x: 2-2*x]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def find_pmap(posterior):
    xs = np.linspace(0,1,1000)
    post = posterior(xs)
    i_max = np.argmax(post)
    return xs[i_max], post[i_max]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
n=1
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, 1, x,left, cs)
alpha=1    
for i in reversed(range(n)):
    pax.plot(xs,posteriors[i](xs), zorder=1, c='blue', alpha=alpha);
    alpha*=0.75
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map, y_map = find_pmap(posteriors[n-1])

plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

### Maximal a posteriori

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
n=1
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, 1, x,left, cs)
alpha=1    
for i in reversed(range(n)):
    pax.plot(xs,posteriors[i](xs), zorder=1, c='blue', alpha=alpha);
    alpha*=0.75
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map, y_map = find_pmap(posteriors[n-1])
pax.annotate(f'MAP $p={p_map:.1f}$',(p_map, y_map),(p_map+0.2, y_map), 
             fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
plt.close();
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
nb=2;n=1
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, nb, x,left, cs)
alpha=1    
for i in reversed(range(n)):
    pax.plot(xs,posteriors[i](xs), zorder=1, c='blue', alpha=alpha);
    alpha*=0.75
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map = 0.0;y_map = posteriors[0](p_map);
pax.annotate(f'MAP $p={p_map:.1f}$',(p_map, y_map),(p_map+0.2, y_map), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$$P(p|r,r) = \frac{P(r|p) P(p|r)}{\int_0^1\text{d}p\,P(r|p) P(p|r)}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(r|p) P(p|r)=2(1-p)^2,\quad 2\int_0^1\text{d}p (1-p)^2=\frac{2}{3}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(p|r,r)=3(1-p)^2$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
posteriors.append(lambda x: 3*(1-x)**2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
nb=2;n=2
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, nb, x,left, cs)
alpha=1    
for i in reversed(range(n)):
    pax.plot(xs,posteriors[i](xs), zorder=1, c='blue', alpha=alpha);
    alpha*=0.75
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map, y_map = find_pmap(posteriors[n-1])
pax.annotate(f'MAP $p={p_map:.1f}$',
             (p_map, y_map),(p_map+0.2, y_map-0.25), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
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
nb=3;n=2
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, nb, x,left, cs)
alpha=1    
for i in reversed(range(n)):
    pax.plot(xs,posteriors[i](xs), zorder=1, c='blue', alpha=alpha);
    alpha*=0.75
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map = 0.0;y_map = posteriors[n-1](p_map);
pax.annotate(f'MAP $p={p_map:.1f}$',(p_map, y_map),(p_map+0.2, y_map-0.25), 
             fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

$$P(p|n_l,n_r)=\frac{P(n_l,n_r|p)P(p)}{P(n_l,n_r)}$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Binomial distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n_l,n_r|p) = \binom{n_l+n_r}{n_l}p^{n_l}(1-p)^{n_r}$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Likelihood

+++ {"slideshow": {"slideshow": {"slide_type": "fragment"}}, "tags": []}

$$L(p|n^0, n^1) = P(n^0,n^1|p) = \binom{n^0+n^1}{n^1}p^{n^1}(1-p)^{n^0}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Maximal likelihood

+++ {"slideshow": {"slide_type": "skip"}}

Now we can choose $p$ as the number that maximises this quantity. Actually when dealing with probabilities it's often more convenient to use logarithms. Logarithm is a monotonicaly increasing funcion, so the maximum of logarithm of an function will correspond to maximum of the function.

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$\log P(n^0,n^1|p) = \log\binom{n^0+n^1}{n^1}+\log p^{n^1}+\log (1-p)^{n^0} =  \log\binom{n^0+n^1}{n^1}+n^1\log p+{n^0}\log (1-p)$$

+++ {"slideshow": {"slide_type": "skip"}}

Differentiating this expression with respect to $p$ we obtain equation for minimum:

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{n^1}{p}-\frac{n^0}{1-p}=0$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\no(1-p) = \nz p$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\no = \nz p +\no p$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$p=\frac{\no} {\no+\nz} $$

+++ {"slideshow": {"slide_type": "skip"}}

Leading to the result stated above.

+++ {"slideshow": {"slide_type": "notes"}}

This approach works, the estimator is _consistent_, that is as the number of tosses goes to infinity its value will converge to the true value  of $p$. However it has some problems.


 

First of all the likelihood has no clear interpretation. This is NOT a probability distribution on $p$ ! Ideally we would like to have the distribution

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

$$P(p|n^1, n^0)$$

+++ {"slideshow": {"slide_type": "notes"}}

which can be interpreted as probability of $p$ given the measured $\nz$ amd $\no$.

+++ {"slideshow": {"slide_type": "notes"}}

Secondly with maximal likelihood estimator there is no clear way of estimating the error on the $p$.

+++ {"slideshow": {"slide_type": "notes"}}

And thirdly we do not have a way of incorporating our prior knowledge. For example when tossing a coin we are practically sure that $p=1/2$. If we toss the coin and get  values

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p)=1 $$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(p|n_l,n_r)=\frac{P(n_l,n_r|p)P(p)}{P(n_l,n_r)}=\frac{P(n_l,n_r|p)P(p)}{\int_0^1\text{d}p' P(n_l,n_r|p')P(p')}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\int_0^1\text{d}p' P(n_l,n_r|p')P(p') = \binom{n_l+n_r}{n_l}
\frac{n_l!n_r!}{(n_l+n_r+1)!}=\frac{1}{n_l+n_r+1}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p|n_l,n_r)=\binom{n_l+n_r}{n_l}(n_l+n_r+1)p^{n_l}(1-p)^{n_r}$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

### Beta distribution

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(x|\alpha,\beta) =  \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
x^{\alpha-1}(1-x)^{\beta-1}
$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$p^{n_l}(1-p)^{n_r} \sim \operatorname{Beta}(n_l+1,n_r+1) $$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import beta
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
posteriors=[]
for i in range(1,101):
    n_l = left[:i].sum()
    n_r = i-n_l
    posteriors.append(beta(n_l+1, n_r+1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
nb=3;n=3
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, nb, x,left, cs)
alpha=1    
for i in reversed(range(n)):
    pax.plot(xs,posteriors[i].pdf(xs), zorder=1, c='blue', alpha=alpha);
    alpha*=0.75
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
p_map = n_l/(n_l+n_r); y_map=posteriors[n-1].pdf(p_map)
pax.annotate(f'MAP $p={p_map:.2f}$',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
```

+++ {"slideshow": {"slide_type": "slide"}}

### Confidence interval

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
from scipy.optimize import fsolve
def confidence_interval(dist,center,alpha):
    cdf = dist.cdf
    f = lambda x: cdf(center)-cdf(x)-alpha/2
    left = fsolve(f,center)  
    f = lambda x: cdf(x)-cdf(center)-alpha/2
    right = fsolve(f,center)
    return left.item(), right.item()
    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
nb=3;n=3
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
#plot_balls(ax, nb, x,left, cs)
alpha=1
pax.plot(xs,posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
post = posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
l,r=confidence_interval(post, p_map,0.75)
pax.fill_between(xs,posteriors[2].pdf(xs),0, where = (xs>l) & (xs<r));
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Mean

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
nb=3;n=3
fig,ax,pax = make_pool_figure(ylim=(0,3))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
#plot_balls(ax, nb, x,left, cs)
alpha=1
pax.plot(xs,posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);

pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
p_map = n_l/(n_l+n_r); y_map=posteriors[n-1].pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
p_mean = posteriors[n-1].mean(); p_std = posteriors[n-1].std();
pax.axvline(p_mean);
pax.fill_between(xs,posteriors[2].pdf(xs),0, where = (xs> p_mean-p_std) & (xs<p_mean+p_std));
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
nb=100;n=100
fig,ax,pax = make_pool_figure(ylim=(0,10))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, nb, x,left, cs, draw_line=False, alpha=0.7)
alpha=1    
pax.plot(xs,posteriors[99].pdf(xs), zorder=1, c='blue', alpha=alpha);
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
n_l = left[:n].sum(); n_r = n- n_l;
post = posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
l,r=confidence_interval(post, p_map,0.90)
pax.fill_between(xs,post.pdf(xs),0, where = (xs>l) & (xs<r), alpha=0.5);
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

+++ {"slideshow": {"slide_type": "slide"}}

### Highest density region (HDR)

+++

$\beta$ HDR is a region where at least $\beta$ of probability is concentrated and has smallest possible volume in the sample space, hence highest density. More formal definition given below.

+++ {"slideshow": {"slide_type": "-"}}

Let $P_X(p)$ be de density function of  some random variable $X$ with values in $R_X$. Let' $R_X(p)$ be the subsets of $R_X$ such  that

+++ {"slideshow": {"slide_type": "-"}}

$$ R(p) = \{x\in R_X: P_X(x)\ge p\}$$

+++ {"slideshow": {"slide_type": "-"}}

The $\beta$ HDR is equal to $R(p_\beta)$ where $p_\beta$ is the largest constant such that

+++ {"slideshow": {"slide_type": "-"}}

$$P\left(x\in R(p_\beta)\right)\ge \beta$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
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
  slide_type: skip
---
nb=100;n=100
fig,ax,pax = make_pool_figure(ylim=(0,10))
plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
plot_balls(ax, nb, x,left, cs, draw_line=False, alpha=0.7)
alpha=1    
pax.plot(xs,posteriors[99].pdf(xs), zorder=1, c='blue', alpha=alpha);
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
n_l = left[:n].sum(); n_r = n- n_l;
post = posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
l,r= HDIofICDF(beta, 0.95, a=n_l+1, b= n_r+1)
pax.fill_between(xs,post.pdf(xs),0, where = (xs>l) & (xs<r), alpha=0.5);
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

## Female births

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
f_births = 241945  
m_births = 251527
```

+++ {"slideshow": {"slide_type": "fragment"}}

$$ P\left(p_f>=\frac{1}{2}\right) = \frac{\int_{\frac{1}{2}}^1 p_f^{241945}(1-p_f)^{251527}\text{d}p_f}
{\int_{0}^1  p_f^{241945}(1-p_f)^{251527}\text{d}p_f} \approx 1.15 \times 10^{-42}$$

+++ {"slideshow": {"slide_type": "slide"}}

## Incremental information

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$Beta(n_l+1+1,n_r+1)\sim p^{n_l+1} (1-p)^{n_r} \sim p Beta(n_l+1,n_r+1) \sim P(l|p) Beta(n_l+1,n_r+1) $$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$Beta(n_l+1,n_r+1+1)\sim p^{n_l} (1-p)^{n_r+1} \sim (1-p) Beta(n_l+1,n_r+1) \sim P(r|p) Beta(n_l+1,n_r+1) $$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(\theta|D_2, D_2y1)=\frac{P(D_2,D_1|\theta)P(\theta)}{P(D_2, D_1)}=\frac{P(D_2|\theta)P(D_1|\theta)P(\theta)}{P(D_2,D_1)}\propto P(D_2|\theta)P(\theta|D_1)$$

+++ {"slideshow": {"slide_type": "slide"}}

## Conjugate  priors

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p)$$

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(p)\sim Beta(\alpha,\beta)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: []
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,2,5,10]:
    ys = st.beta(a,a).pdf(xs)
    plt.plot(xs,ys, label='%4.2f' %(a,))
plt.legend(loc=1);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
ys = st.beta(0.2,3).pdf(xs)
plt.plot(xs,ys);
```

+++ {"slideshow": {"slide_type": "notes"}}

It can be more convenient to parametrise  Beta distribution by its mean and variance. The mean and variance of Beta distribution are

+++ {"slideshow": {"slide_type": "slide"}}

$$\mu = \frac{\alpha}{\alpha+\beta}\quad\text{and}\quad \sigma^2=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

+++ {"slideshow": {"slide_type": "skip"}}

and so

+++ {"slideshow": {"slide_type": "fragment"}}

$$\nu = \frac{\mu(1-\mu)}{\sigma^2}-1,\quad \alpha = \mu\nu, \quad
\beta = (1-\mu) \nu,\quad \sigma^2<\mu(1-\mu)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [function]
---
def beta_mu_var(mu, s2):
    """Returns Beta distribution object (from scipy.stats) with specified mean and variance"""
    
    nu = mu*(1-mu)/s2 -1
    if nu>0:
        alpha = mu*nu
        beta = (1-mu)*nu
        return st.beta(a=alpha,b=beta)
    else:
        print("s2 must be less then {:6.4f}".format(mu*(1-mu)))
```

+++ {"slideshow": {"slide_type": "slide"}}

### Back to coin toss

+++ {"slideshow": {"slide_type": "fragment"}}

So let's assume that the $p$ values of coins produced by our sloppy blacksmith have Beta distribution  with mean $\mu=0.45$ and standard deviation $\sigma=0.1$.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
prior = beta_mu_var(0.45, 0.1*0.1)
pars = prior.kwds
alpha =pars['a']
beta = pars['b']
print("alpha = {:.2f}, beta={:.2f}".format(alpha,beta))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
uni_prior = st.beta(1,1)
```

+++ {"slideshow": {"slide_type": "notes"}}

We will compare this to uniform prior with $\alpha=\beta=1$. This gives a constant probability density function $P(p)=1$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,2000)
plt.plot(xs, prior.pdf(xs),    label="$\\alpha = {:5.2f}$ $\\beta = {:5.2f}$".format(alpha, beta));
plt.plot(xs,uni_prior.pdf(xs), label="$\\alpha = {:5.2f}$ $\\beta = {:5.2f}$".format(1, 1));
plt.xlabel('p');
plt.ylabel('P(p)');
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

### Posterior

+++ {"slideshow": {"slide_type": "fragment"}}

$\newcommand{\nz}{n^{0}}\newcommand{\no}{n^{1}}$
$$ P(p\nz,\no) =\frac{P(\nz,\no|p)P(p)}{P(\nz,\no)},\qquad P(\nz,\no) = \int\text{d}p\,P(\nz,\no|p)P(p)$$

+++ {"slideshow": {"slide_type": "notes"}}

If we chose $Beta(\alpha, \beta)$ as the prior $P(p)$ the  numerator of the posterior distribution has the form:

+++ {"slideshow": {"slide_type": "fragment"}}

$$ \binom{\nz+\no}{\no}p^{\no}(1-p)^{\nz}  \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
p^{\alpha-1} (1-p)^{\beta-1}$$

+++ {"slideshow": {"slide_type": "notes"}}

The denominator is an integral of the numerator but we don't have to do it explicitely. By noting that

+++ {"slideshow": {"slide_type": "slide"}}

### Conjugate priors

+++ {"slideshow": {"slide_type": "fragment"}}

$$ p^{\no}(1-p)^{\nz} p^{\alpha-1}(1-p)^{\beta-1}=
p^{\no+\alpha-1}(1-p)^{\nz+\beta-1}$$

+++ {"slideshow": {"slide_type": "notes"}}

we see that the functional dependence on $p$ is same as for $Beta(n^1+\alpha, n^0+\beta)$. So this is the same distribution and  finally we have the formula for the posterior

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p| \nz, \no )  \sim  Beta(\no+\alpha, \nz+\beta)$$

+++ {"slideshow": {"slide_type": "skip"}}

So continuing our example let's suppose that real $p$ of our coin is

+++ {"slideshow": {"slide_type": "notes"}}

A prior with a property that the posterior distribution has same form as the prior is called _conjugate_ prior to the sampling distribution. So the Beta distribution is a conjugate prior to Bernouilli distribution.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p_coin = 0.35
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
coin = st.bernoulli(p=p_coin)
```

+++ {"slideshow": {"slide_type": "skip"}}

We will "toss" it 10000 times

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
n_tosses = 10000
tosses = coin.rvs(n_tosses)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [function]
---
def ht(tosses):
    """Takes a list of toss results and returns number of successes and failures"""
    h = tosses.sum()
    t = len(tosses)-h
    return (h,t)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ht(tosses)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [function]
---
def ab_string(a,b):
    return f"$\\alpha = {a:.2f}$ $\\beta = {b:.2f}$"

def draw_beta_prior(a,b,**kargs):
    xs=np.linspace(0,1,1000)
    plt.plot(xs, st.beta(a,b).pdf(
        xs), **kargs)

def draw_beta_posterior(a,b,tosses, **kargs):
    """Draw posterior distribution after  seing tosses assuming Beta(a,b) prior"""
    (h,t)=ht(tosses)
    xs=np.linspace(0,1,1000)
    plt.plot(xs, st.beta(a+h,b+t).pdf(xs), **kargs)
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's draw the posterior after 10 tosses

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
draw_beta_posterior(alpha,beta,tosses[:10], label=ab_string(alpha, beta))
plt.legend();
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's discuss again what does this probability distribution mean?

+++ {"slideshow": {"slide_type": "slide"}}

You can thing about it as an outcome of following experiment:
 1. You draw a value for $p$  from the prior distribution
 1. You draw 10 ten times from the Bernoulli distribution with $p$ selected above,  which is equivalent to drawing from Binomial distribution with same $p$  and $n=10$.
 1. You  repeat the two points above noting each time  $p$ and number of successes.
 1. From the results you select only those where number of successes was equal to `tosses[:10].sum()`
 1. The distributiion of $p$ in this selected results should match our posterior!
 
 Let's check this.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def experiment(prior,n,size):
    p = prior.rvs(size=size)
    return np.stack((p,st.binom(n=n, p=p).rvs()), axis=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
res = experiment(st.beta(alpha, beta),10,1000000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(res[res[:,1]==tosses[:10].sum()][:,0], bins=50, density=True, label=ab_string(alpha, beta));
draw_beta_posterior(alpha,beta,tosses[:10], label=ab_string(alpha, beta))
draw_beta_posterior(3,3,tosses[:10], label=ab_string(3, 3))
draw_beta_posterior(1,1,tosses[:10], label=ab_string(1, 1))
plt.legend(title = 'Priors');
```

+++ {"slideshow": {"slide_type": "notes"}}

As we can see we indeed get the predicted posterior distribution. Unfortunatelly this requires us to get our prior right.

+++ {"slideshow": {"slide_type": "notes"}}

With more data the dependence of the posterior on the prior diminishes

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
res_100 = experiment(st.beta(alpha, beta),100,1000000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(res_100[res_100[:,1]==tosses[:100].sum()][:,0], bins=50, density=True);
draw_beta_posterior(alpha,beta,tosses[:100], label=ab_string(alpha, beta))
draw_beta_posterior(3,3,tosses[:100], label=ab_string(1,1), c='red')
draw_beta_posterior(1,1,tosses[:100], label=ab_string(1,1))
plt.legend();
```

+++ {"slideshow": {"slide_type": "notes"}}

So let's see how the posterior distribution evolves with increasing number of tosses. Below we draw posterior distribution after different number of tosses

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
for n in [0, 1,2,3,4,5,10,20,50,100]:
    draw_beta_posterior(alpha,beta,tosses[:n], label="{:d}".format(n))
plt.legend();
plt.axvline(p_coin);
```

+++ {"slideshow": {"slide_type": "skip"}}

And below after some more tosses

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.xlim(0.2,0.5)
for n in [100,200,500,1000,5000]:
    draw_beta_posterior(alpha,beta,tosses[:n], label="{:d}".format(n))
plt.legend();
plt.axvline(p_coin);
```

+++ {"slideshow": {"slide_type": "notes"}}

Let's compare  how the estimated value converges to the real one for different priors. We will use the maximal a posteriori estimate of $p$

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

#### MAP (Maximal a posteriori)

+++ {"slideshow": {"slide_type": "skip"}}

Because mode of the Beta distribution is

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$\frac{\alpha-1}{\alpha+\beta-2},\qquad \alpha, \beta>1$$

+++ {"slideshow": {"slide_type": "skip"}}

the mode of posterior is:

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$ p_{MAP} = \frac{\alpha-1+\no}{\alpha-1+\no+\beta-1+\nz}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
cs  = np.cumsum(tosses)
ns  = np.arange(1.0, len(cs)+1)
avs = cs/ns
post_avs = (cs + alpha-1)/(ns+alpha+beta -2 )
```

+++ {"slideshow": {"slide_type": "notes"}}

So adding a Beta prior amounts to adding $\alpha-1$  and $\beta-1$  repectively to  $\no$ and $\nz$.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
l = 500
plt.plot(ns[:l],avs[:l],'.', label='uniform prior');
plt.plot(ns[:l],post_avs[:l],'.', label='prior');
plt.axhline(p_coin, linewidth=1, c='grey')
plt.legend();
```

+++ {"slideshow": {"slide_type": "notes"}}

We can see that after few tens/houndreds of tosses both estimate behave in the same way, but with informative prior  we get better results for small number of tosses.

+++ {"slideshow": {"slide_type": "slide"}}

#### Problem

+++ {"slideshow": {"slide_type": "-"}}

Assume that the coin is fair _.i.e._ the prior has mean equal 1/2 and standard deviation 1 /50. We after $n$ tosses we get $n$ heads. 

How big must $n$ be so  that the posterior probability $P(p>0.75|n,0)$ is greater then 10% ?

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fair_prior = beta_mu_var(0.5, (1/50)**2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ps = np.linspace(0,1,500)
plt.plot(ps,fair_prior.pdf(ps));
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fair_a = fair_prior.kwds['a']
fair_b = fair_prior.kwds['b']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def fair_posterior_p(n):
        d = st.beta(a=fair_a+n, b=fair_b)
        return 1-d.cdf(0.75)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
for n in range(1,1000):
    P= fair_posterior_p(n)
    if P >0.1:
        break
print(n,P)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
d = st.beta(a=fair_a+n, b=fair_b)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
plt.plot(ps,d.pdf(ps));
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
d.cdf(0.75)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
plt.axvline(0.75,c='grey', linewidth=1)
plt.axhline(0.9, c='grey', linewidth=1)
plt.plot(ps,d.cdf(ps));
```

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

### Posterior predictive distribution

+++ {"slideshow": {"slide_type": "notes"}}

We can ask what is the probability of the coin comming head up after seing it come head up $\no$ times in $n$ trials? The answer is the integral

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$
P(X=1|\no,\nz)=\int\limits_0^1\text{d}p \,P(X=1|p) P(p|\no,\nz) = \int\limits_0^1\text{d}p\, p\, P(p|\no,\nz)
$$

+++ {"slideshow": {"slide_type": "skip"}}

which is an expectation value (mean) of the posterior  distribution leading to:

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$P(X=1|\no,\nz)=\frac{\alpha+\no}{\alpha+\no+\beta+\nz}$$

+++ {"slideshow": {"slide_type": "notes"}}

With uniform prior we obtain the so called

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

#### Laplace Rule of succession

+++ {"slideshow": {"slide_type": "-"}}

The probability of succes after seing $\no$ successes and $\nz$ failures  is

+++ {"slideshow": {"slide_type": "-", "slideshow": {"slide_type": "fragment"}}}

$$P(succes) =  \frac{\no+1}{\no+\nz +2}$$

+++ {"slideshow": {"slide_type": "notes"}}

This is also known as  _Laplace smoothing_.

+++ {"tags": ["problem"], "slideshow": {"slide_type": "slide"}}

__Problem__  Amazon reviews

+++ {"slideshow": {"slide_type": "-", "slideshow": {"slide_type": "-"}}, "tags": ["problem"]}

You can buy same item from two  sellers one with 90 positive  and 10 negative reviews and another with 6 positive  and no negative reviews.
From which you should buy ? What assumption you had to make?

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

Let's assume that for each seller sale is  an independent Bernoulli trial with success denoting no problems for the buyer. The other assuption that we are going to make is that all buyers write the reviews.  If so then by the rule of succession probability of success for the first buyer on the next deal is

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
(90+1)/(100+2)
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

and for the second

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
(6+1)/(6+2)
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

We should buy from the first.

+++ {"slideshow": {"slide_type": "slide"}}

## Categorical variables

+++ {"slideshow": {"slide_type": "notes"}}

A natural generalisation of the Bernoulli distribution is the multinouilli or categorical distribution and the generilsation of the binomial distribution is the _multinomial_ distribution.

+++ {"slideshow": {"slide_type": "notes"}}

Let's say we have $m$ categories with probability $p_k$ for each category. Then after $n$ trials the probability that we $n_k$ results in category $k$ is:

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n_1,\ldots, n_{m}|p_1,\ldots, p_{m}) = \frac{N!}{n_1!\cdots n_{m}!}p_1^{n_1}\cdots p_{m}^{n_{m}}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Example: Dice

+++ {"slideshow": {"slide_type": "-"}}

$$m=6,\quad p_i=\frac{1}{6} \qquad P(n_1,\ldots, n_{6}) = \frac{N!}{n_1!\cdots n_{m}!}\frac{1}{6^N}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Dirichlet distribution

+++ {"slideshow": {"slide_type": "notes"}}

Conjugate prior  to this distribution is the Dirichlet distribution which is a generalisation of the Beta distribution. It has $m$ parameters $\alpha_k$ and its probability mass function is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{Dir}(p_1,\ldots,p_{m}|\alpha_1,\ldots,\alpha_{m}) = \frac{\Gamma\left(\sum\limits_{i=1}^{m} \alpha_i\right)}{\prod\limits_{i=1}^{m}\Gamma(\alpha_i)}
\prod\limits_{i=1}^{m}p_i^{\alpha_i-1}$$

+++ {"slideshow": {"slide_type": "slide"}}

#### Posterior

+++ {"slideshow": {"slide_type": "skip"}}

It is easy to check that the posterior on $p_k$ density is given by the  Dirichet distribution with paramerters $\alpha_1+n_1,\ldots, \alpha_{m}+n_{m}$.

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p_1,\ldots, p_{m}|n_1,\ldots, n_{m})=P_{Dir}(p_1,\ldots,p_{m}|\alpha_1+n_1,\ldots,\alpha_{m}+n_m)$$

+++ {"slideshow": {"slide_type": "notes"}}

The maximal a posteriori estimate is:

+++ {"slideshow": {"slide_type": "fragment"}}

$$p_k = \frac{n_k+\alpha_k-1}{n + \sum_i \alpha_k-m}$$

+++ {"slideshow": {"slide_type": "skip"}}

and Laplace smoothing takes the form:

+++ {"slideshow": {"slide_type": "fragment"}}

$$p_k = \frac{n_k+1}{\sum_{k=1}^m n_k  + m}$$

+++ {"slideshow": {"slide_type": "slide"}}

You can learn more _.e.g._ from ["Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers).P

+++ {"slideshow": {"slide_type": "slide"}}

### Poisson distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y|\theta) = e^{-\theta}\frac{\theta^y}{y!}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y|\theta) = \prod_i e^{-\theta}\frac{\theta^y_i}{y_i!}\propto \theta^{t(y)}e^{-n\theta},\quad t(y)=\sum_i y_i$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta)\propto e^{-\beta\theta}\theta^{\alpha-1},\quad Gamma(\alpha,\beta)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\theta|y \sim Gamma(\alpha+n \bar y,\beta+n)$$
