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
---
import numpy as np
import scipy as sp
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}}

# Bayesian Data Analysis

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\theta|y) \propto P(y|\theta) P(\theta)$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

## Nuisance parameters

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$P(\theta_1,\theta_2|y) \propto P(y|\theta_1,\theta_2) P(\theta_1,\theta_2)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1|y) = \int\text{d}{\theta_2}P(\theta_1,\theta_2|y) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1|y) = \int\text{d}\theta_2 P(\theta_1|y,\theta_2)P(\theta_2|y)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model with know variance

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y|\mu,\sigma) =\prod_k \frac{1}{\sqrt{2\pi}\sigma} 
e^{-\frac{1}{2\sigma^2}\left(y_k-\mu\right)^2}$$

+++ {"slideshow": {"slide_type": "slide"}}

###  Uninformative (improper) prior

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|y,\sigma)=g(y-\mu,\sigma)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$g(y-\mu|\sigma)P(\mu) \propto f(y-\mu,\sigma)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu)\propto 1$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu|y,\sigma) \propto \prod_k \frac{1}{\sqrt{2\pi}\sigma} 
e^{-\frac{1}{2\sigma^2}\left(y_k-\mu\right)^2}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu|y,\sigma) \propto  \left(\sqrt{2\pi}\sigma\right)^{-n} 
e^{-\frac{1}{2\sigma^2}\sum_{k=1}^n\left(y_k-\mu\right)^2}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu|y,\sigma) \propto  \sigma^{-n} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n}{2\sigma^2}\left(\overline{y^2} -{\bar y }^2\right)}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\mu|y,\sigma \sim \operatorname{Norm}\left(\bar y,\frac{\sigma}{\sqrt{n}}\right)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Uknow variance

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y|\mu,\sigma)=\frac{1}{\sigma}\cdot g\left(\frac{y-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu,\sigma|y) = \frac{y}{\sigma^2} f\left(\frac{y-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{1}{\sigma}g\left(\frac{y-\mu}{\sigma}\right)P(\mu,\sigma)\propto \frac{y}{\sigma^2}f\left(\frac{y-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\sigma)\propto \frac{1}{\sigma},\quad P(\sigma^2)\propto \frac{1}{\sigma^2}$$

+++ {"slideshow": {"slide_type": "slide"}}

$$
P(\mu,\sigma^2|y) \propto  \sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n}{2\sigma^2}\left(\overline{y^2} -{\bar y }^2\right)}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu,\sigma^2|y) \propto  (\sigma^2)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2},\quad s^2=\frac{n}{n-1}\left(\overline{y^2} -{\bar y }^2\right)
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\sigma^2|y) \propto  \int\text{d}\mu\,\sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}
\propto
\sigma^{-n-2} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}\sqrt{2\pi\frac{\sigma^2}{n}}
$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\left(\sigma^2\right)^{-\frac{n+1}{2}} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sigma^2|y \sim \operatorname{Scale-Inv-}\chi^2(n-1,s^2)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu,\sigma^2|y) = P(\mu|\sigma^2,y)P(\sigma^2|y)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu|y)=\int\text{d}\sigma^2 P(\mu,\sigma^2|y) \propto \int_0^\infty\text{d}\sigma^2 \sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$z=\frac{A}{2\sigma^2},\quad A=n(\bar y-\mu)+(n-1)s^2$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\text{d}z=-\frac{A}{2\sigma^4}\text{d}\sigma^2$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|y)\propto A^{-\frac{n}{2}}\int_0^\infty\text{d}z\,z^{\frac{n-1}{2}}e^{-z}$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\left.
\frac{\mu-\bar y}
{\sqrt{\frac{s^2}{n}}}
\right|\sim t_{n-1}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Signal with background

> "Data Analysis, A Bayesian Tutorial" D.S.Sivia, J. Skilling

+++ {"slideshow": {"slide_type": "slide"}}

$$D_k = n_0 \left(A\, e^{\displaystyle -\frac{1}{2w^2}(x_k - x_0)^2 } +B \right)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Poisson distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n|D) = e^{-D}\frac{D^n}{n!} $$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(D|n)=\prod_{k=1}^M e^{-D}\frac{D^{\displaystyle n_k}}{n_k!}\propto e^{\displaystyle-M D+\log D\sum_{k=1}^M n_k}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(D|n)\propto e^{\displaystyle-M D+\log D\sum_{k=1}^M n_k}=
e^{\displaystyle-M D} D^{\displaystyle \sum_k n_k}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$D|n\sim \operatorname{Gamma}\left(1+\sum_k n_k,\frac{1}{M}\right)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
lb = 5.77
M=7
nk_pois = st.poisson(lb).rvs(size=M) 
print(nk_pois)
nk_sum = nk_pois.sum()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
xs = np.linspace(0,20,200)
ys = st.gamma(nk_sum,scale=1./M).pdf(xs)
plt.plot(xs,ys)
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\{n_k\}|A,B)=\prod_{k=1}^M P(n_k|A,B)=\prod_{k=1}^M P(n_k|D_k(A,B)) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A,B|\{n_k\})=P(A,B)\prod_{k=1}^M P(n_k|A,B)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(A,B) = \begin{cases}1 & 
A\ge0, B\ge0\\
0 & \text{otherwise}
\end{cases}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log P(A,B|\{n_k\})= \log P(A,B) +\sum_k \log P(n_k|A,B)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log P(n_k|A,B)=n_k \log D_k-D_k -\log( n_k!)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
w  = 2.12
x0 = 0
A_true = 1
B_true = 2
n0=32
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def D_rate(x,A,B):
    b = 1.0/(2*w**w)
    return n0*(np.multiply.outer(np.exp(-b*(x-x0)*(x-x0)),A)+B)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
D_rate(np.array([1,2,3]),np.array([1,2]),np.array([1,2]) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xk = np.arange(-6,7)
dk = D_rate(xk,A_true, B_true) 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
nk = st.poisson(dk).rvs(size=len(dk))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.step(xk,dk, where='mid')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(xk,nk, drawstyle='steps-mid')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def log_pk(A,B):
    dk = D_rate(xk,A,B)
    dim = len(dk.shape)
    sh=np.ones(dim).astype('int')
    sh[0]=-1
    lk =  np.log(dk)*nk.reshape(sh) - dk
    return lk

def log_p(A,B):
    lk = log_pk(A,B)
    return lk.sum(axis=0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
As = np.linspace(0.25,2,100)
Bs = np.linspace(1.25,3,100)
xs, ys = np.meshgrid(As,Bs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
zs = log_p(xs,ys)
nzs=zs-np.max(zs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
idx = np.unravel_index(np.argmax(zs), zs.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(xs[idx], ys[idx])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.contourf(xs,ys,nzs, levels=np.log(np.array([0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])))
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(As,np.exp(nzs).sum(axis=0))
plt.grid()
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(Bs,np.exp(nzs).sum(axis=1))
plt.grid()
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Ellection polls

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
poll = {'PiS':0.424, 
        'PlatformaObywatelska':0.30, 
        'SLD':0.074, 'Kukiz15': 0.073, 
        'PSL':0.06, 
        'Razem':0.024, 
        'PartiaWolności': 0.019 }
```

```{code-cell} ipython3
prob = np.asarray(list(poll.values()))
responses = np.array(1017*prob, dtype='int')
```

```{code-cell} ipython3
n=np.sum(responses)
probs = responses/n
```

+++ {"slideshow": {"slide_type": "slide"}}

### Multinomial distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n_1,\ldots,n_k|p_1,\ldots,p_k)=\frac{(\sum_{i=1}^k n_i )!}{n_1!\cdots n_k!}p_1^{n_1}\cdots p_k^{n_k}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p_1,\ldots,p_k|\alpha_1,\dots,\alpha_k)=\frac{\Gamma(\sum_{i=1}^k \alpha_i)}{\Gamma(\alpha_1)
\cdots \Gamma(\alpha_k)}p_1^{\alpha_1-1}\cdots p_k^{\alpha_k-1}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p_1,\ldots,p_k|n_1,\ldots,n_k,\alpha_1,\ldots,\alpha_k)\propto P(n_1,\ldots,n_k|p_1,\ldots,p_k)P(p_1,\ldots,p_k|\alpha_1,\dots,\alpha_k)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$p_1,\ldots,p_k|n_1,\ldots,n_k,\alpha_1,\ldots,\alpha_k\sim \operatorname{Dirichlet}(\alpha_1+n_1,\ldots,\alpha_k+n_k)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sample =  st.dirichlet(alpha=responses+1).rvs(size=(100000))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample[:,0], bins=100, histtype='step', density=True)
plt.hist(sample[:,1], bins=100, histtype='step', density=True)
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
h = ax.hist2d(sample[:,0], sample[:,1], bins=100)
plt.colorbar(h[3],ax=ax)
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
h = ax.hexbin(sample[:,0], sample[:,1], bins=100)
plt.colorbar(h,ax=ax)
plt.grid()
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample[:,4], bins=100, histtype='step', density=True)
plt.axvline(0.05)
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample[:,3]-sample[:,4], bins=100, histtype='step', density=True)
plt.axvline(0.0)
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Logistic regression

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
dose = np.array([-0.86,-0.3, -0.05, 0.73])
n_animals = np.array([5,5,5,5])
n_deaths = np.array([0,1,3,5])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(dose, n_deaths/n_animals,'o')
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

$$y_i|\theta_i \sim \operatorname{Bin}(n_i,\theta_i)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$\newcommand{\logit}{\operatorname{logit}}$
$$\logit(\theta) = \alpha+\beta x_i $$

+++ {"slideshow": {"slide_type": "slide"}}

$$\logit(\theta)=\log\frac{\theta}{1-\theta}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\theta = \frac{e^{\logit\theta}}{1+e^{\logit\theta}}\equiv s\left(\logit(\theta)\right)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log s(x) = x-\log(1+e^x)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\alpha,\beta|y,n,x)=P(y|\alpha,\beta,n,x)p(\alpha,\beta|n,x)=P(\alpha,\beta)\prod_k P(y_i|\alpha,\beta,n_i,x_i)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y_i|\alpha,\beta,n_i,x_i)\propto s(\alpha+\beta x_i)^{y_i}(1-s(\alpha+\beta x_i))^{n_i-y_i}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\alpha,\beta)\propto 1$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def s(x):
    return 1.0/(1+np.exp(-x))

def ls(x):
    return x-np.log(1+np.exp(x))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def log_P_alpha_beta(a,b):
    lg = a + np.multiply.outer(dose,b)
    ltheta = ls(lg)
    ltheta_conj = -np.log(1+np.exp(lg))
    sh = np.ones(len(ltheta.shape),dtype='int')
    sh[0] = -1
    return -np.sum((ltheta*n_deaths.reshape(sh))+
                   ((ltheta_conj)*(n_animals-n_deaths).reshape(sh)), axis=0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
opt.minimize(lambda arg: log_P_alpha_beta(*arg),[10,10])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
alphas = np.linspace(-1,5, 100)
betas  = np.linspace(-1,25,100)
a_mesh, b_mesh = np.meshgrid(alphas, betas)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
lr_zs =  log_P_alpha_beta(a_mesh, b_mesh)
lr_zs= lr_zs - lr_zs.min()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.contourf(alphas, betas, lr_zs, levels= -np.log([1,0.9, 0.7, 0.5, 0.3,0.1,0.01]))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p_alpha_beta=np.exp(-lr_zs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
p_alphas  = p_alpha_beta.sum(axis=0)
p_alphas /= p_alphas.sum()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(alphas, p_alphas)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
alphas_dist = st.rv_discrete(0,len(p_alphas)-1,
               values=( range(len(p_alphas)), p_alphas) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
b_dist=np.asarray([st.rv_discrete(0,len(betas)-1,
                values=(
                    range(len(betas)),p_alpha_beta[:,i]/p_alpha_beta[:,i].sum()
                             ))
                        for i in range(len(betas))  ] )
    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def gen(x):
    return x.rvs(size=1)

gen = np.vectorize(gen)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def gen_alpha_beta(n):
    da = (alphas[-1]-alphas[0])/(len(alphas-1))
    db =  (betas[-1]-betas[0])/(len(betas-1))
    ia = alphas_dist.rvs(size=n)
    als = alphas[ia]+st.uniform(loc=da/2,scale=da).rvs(size=n)
    bes = betas[gen(b_dist[ia])]+st.uniform(loc=db/2,scale=db).rvs(size=n)
    return np.stack((als,bes), axis=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ab = gen_alpha_beta(10000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.scatter(ab[:,0], ab[:,1],s=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(ab[:,0], ab[:,1])
```

+++ {"slideshow": {"slide_type": "slide"}}

### LD50 Dose

+++ {"slideshow": {"slide_type": "fragment"}}

$$\theta = \frac{1}{2}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\logit\left(\frac{1}{2}\right)=0$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\alpha +\beta x = 0,\quad x = -\frac{\beta}{\alpha}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(-ab[:,0]/ab[:,1],bins=60, histtype='step')
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
