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

# Monte Carlo methods

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%matplotlib inline
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}}

### Posterior distribution

+++

$$P(\theta|y) \propto P(y|\theta) P(\theta)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\theta_1|y)=\int\text{d}\theta_2 P(\theta_1, \theta_2|y)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1, \theta_2|y)= P(y|\theta_1, \theta_2)P(\theta_1, \theta_2)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\int\text{d}{\theta} f(\theta) P(\theta) = E_{P(\theta)}[f(\theta)]$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$ E_{P(\theta)}[f(\theta)] \approx \frac{1}{N}\sum_{i=1}^N f(\theta^i), \quad \theta^{i}\sim P(\theta)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Generating  random numbers

+++ {"slideshow": {"slide_type": "fragment"}}

#### Generating uniformly distributed random numbers

+++ {"slideshow": {"slide_type": "fragment"}}

#### Generating numbers from a given distribution

+++ {"slideshow": {"slide_type": "slide"}}

### Discrete distributions

+++

$$p_i,\quad \sum_ip_i=1,\quad i=0,\ldots,N-1$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$[0,p_0),[p_0,p_0+p_1),[p_0+p_1,p_0+p_1+p_2),\ldots,[\sum_{i=0}^{N-2}p_i,1)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$[0,c_0), [c_0,c_1),\cdots\qquad  c_i = \sum_{j=0}^i p_j$$

+++ {"slideshow": {"slide_type": "slide"}}

$$u\in [0,1)$$

+++ {"slideshow": {"slide_type": "fragment"}}

```
i=0
while u>c_i:
    i++
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
p = np.array([1/12, 1/3, 1/2, 1/12])
```

```{code-cell} ipython3
cum = np.cumsum(p)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
u = np.random.uniform(0,1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
np.searchsorted(cum,u)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
n_samples =1000
u = np.random.uniform(0,1,n_samples)
```

```{code-cell} ipython3
sample =  np.searchsorted(cum,u)
```

```{code-cell} ipython3
plt.hist(sample,bins=4, range=(-0.5,3.5) , histtype='step')
plt.plot([0,1,2,3], p*n_samples, 'o')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
q = st.norm(5,5).pdf(np.arange(0,20))
q/=q.sum()
```

```{code-cell} ipython3
plt.plot(q, drawstyle='steps-mid')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
cq =  np.cumsum(q)
```

```{code-cell} ipython3
plt.plot(cq, drawstyle='steps-mid')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
u = np.random.uniform(0,1)
i = np.searchsorted(cq,u)
x = cq[i]
ax.plot(cq, drawstyle='steps-post')
ax.plot([0,i],[u,u])
axis_to_data = ax.transAxes + ax.transData.inverted()
a = axis_to_data.inverted().transform([i,u])
ax.axhline(u, xmax=a[0], c='r')
ax.axvline(i, ymax=a[1], c='r')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Inverse cumulant

+++ {"slideshow": {"slide_type": "fragment"}}

$$ CDF(x) = \int_{-\infty}^x\text{d}x' P(x') $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$ \frac{\partial}{\partial x} CDF(x) = P(x)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$x \sim P(x),\quad CDF(x)\sim U(0,1)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$u = CDF(x)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$CDF(u)=P\left(CDF(x)<u\right) = P\left(x<CDF^{-1}(u)\right) \equiv CDF\left(CDF^{-1}(u)\right)=u$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(u)=1$$

+++ {"slideshow": {"slide_type": "slide"}}

$$u \sim U(0,1),\quad CDF^{-1}(u) \sim P(x)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(CDF^{-1}(u)<x) = P(u<CDF(x)) = CDF(x)$$

+++ {"slideshow": {"slide_type": "slide"}}

###  Expotential distribution

+++ {"slideshow": {"slide_type": "-"}}

$$P(x) = \lambda e^{-\lambda x}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$CDF(x) = \lambda\int_{0}^{x}\text{d}{x'}e^{-\lambda x}= -\left. e^{-\lambda x}\right|_{0}^{x}=1-e^{-\lambda x}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$u = 1-e^{-\lambda x}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$x = -\frac{1}{\lambda}\log(1-u) $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
u = np.random.uniform(0,1,100000)
```

```{code-cell} ipython3
x = -np.log(1-u)/2
```

```{code-cell} ipython3
plt.hist(x,bins=50, density=True)
xs = np.linspace(0,10,100)
plt.plot(xs, st.expon(scale=0.5).pdf(xs))
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Markov Chain Monte-Carlo

+++ {"slideshow": {"slide_type": "fragment"}}

$$ P(x^{n}\rightarrow x^{n+1}) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$ \int \text{d}y P(x\rightarrow y) = 1,\quad \int \text{d}x P(x\rightarrow y) >0$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{n+1}(x^{n+1})=\int\text{d}{x^{n}} P_n(x^{n})P(x^{n}\rightarrow x^{n+1})$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y)=\int\text{d}{x} P(x)P(x\rightarrow y)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Detailed balance

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(x)P(x\rightarrow y) = P(y) P(y\rightarrow x)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\int\text{d}{x} P_n(x)P(x\rightarrow y)= \int\text{d}{x} P_n(y)P(y\rightarrow x)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\int\text{d}{x} P_n(y)P(y\rightarrow x) = P(y)\int\text{d}{x} P(y\rightarrow x) = P(y)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Metropolis-Hastings algorithm

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{trial}(x\rightarrow y) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$r(x,y) =  \frac{P(y)P_{trial}(y \rightarrow x)}{P(x)P_{trial}(x \rightarrow y)}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(x\rightarrow y)=P_{trial}(x\rightarrow y) \min\left(1, r(x,y)\right)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(x) P(x\rightarrow y)=P(x) P_{trial}(x\rightarrow y)\min\left(1,r(x,y) \right)$$

+++ {"slideshow": {"slide_type": "slide"}}

$r(x,y)>0$
$$P(x)P(x\rightarrow y)=P(x) P_{trial}(x\rightarrow y)\cdot 1 = P(x) P_{trial}(x\rightarrow y) $$

+++ {"slideshow": {"slide_type": "slide"}}

$r(x,y)>0$
$$P(y)P(y\rightarrow x)=P(y)P_{trial}(y\rightarrow x) r(y,x)= 
P(y  P_{trial}(y\rightarrow x) \frac{P(xP_{trial}(x \rightarrow y)}{P(y)P_{trial}(y \rightarrow x)} = P(x)P_{trial}(x \rightarrow y)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Metropolis algorithm

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{trial}(x\rightarrow y) =P_{trial}(y\rightarrow x) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(x\rightarrow y)=P_{trial}(x\rightarrow y)  \min\left(1, \frac{P(y)}{P(x)}\right)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(x) = e^{\log P(x)}$$

+++ {"slideshow": {"slide_type": "fragment"}}

```
if log P(y) > log P(x):
    x = y
else:
   r = np.random.uniform()
   if r< exp(log P(y) -log P(x):
   x = ty
   
```

+++ {"slideshow": {"slide_type": "slide"}}

### Example  - Normal distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(x|\mu,\sigma)\propto e^{\displaystyle -\frac{(x-\mu)^2}{2\sigma^2}}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log P(x|\mu,\sigma) =  -\frac{(x-\mu)^2}{2\sigma^2} + C$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def log_norm(x,mu=0,s=1):
    return -0.5*(x-mu)*(x-mu)/(s*s)
```

```{code-cell} ipython3
def mcmc(log_p, size,eps=0.1, x = 0):
    chain = [x]
    prev_x = x;
    prev_log_p =  log_p(prev_x)
    for i in range (size):
        trial_x = prev_x + np.random.uniform(-eps,eps)
        trial_log_p = log_p(trial_x)
        accept = True
        if(trial_log_p < prev_log_p):
            r = np.random.uniform(0,1)
            if(r>np.exp(trial_log_p-prev_log_p)):
                accept = False
        if accept:
            prev_x= trial_x
            prev_log_p = trial_log_p
        chain.append(prev_x)
    return np.asarray(chain)       
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time
sigma = 0.5
chain  =  mcmc(lambda x: log_norm(x,0,sigma),1000000,1, 0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain[100:], bins=50, density=True, align='mid')
xs = np.linspace(-4*sigma,4*sigma,100)
plt.plot(xs, st.norm(0,sigma).pdf(xs))
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain[:4000])
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Statistical errors

+++ {"slideshow": {"slide_type": "fragment"}}

$$\bar x = \frac{1}{N}\sum_{i=1}^N x_i$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$E[\bar x] = \frac{1}{N}\sum_{i=1}^N E[x_i] = E[x]$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$Var[\bar x]=E\left[(\bar x-E[\bar x])^2)\right]$$

+++ {"slideshow": {"slide_type": "slide"}}

$$E\left[(\bar x-E[\bar x])^2)\right] = E\left[\frac{1}{N^2}\sum_{ij=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right]$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\begin{split}
E\left[\frac{1}{N^2}\sum_{ij=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right] &=
E\left[\frac{1}{N^2}\sum_{i=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right]\\
&\phantom{=}+
E\left[\frac{1}{N^2}\sum_{i\ne j} \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right]
\end{split}$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{1}{N}Var[x] +
E\left[\frac{1}{N^2}\sum_{i\ne j} \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right] $$

+++ {"slideshow": {"slide_type": "slide"}}

$$\begin{split}
E\left[\frac{1}{N^2}\sum_{i\ne j} \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right] &=\frac{1}{N^2}\sum_{i\ne j}corr(x_i,x_j)\\
& = \frac{1}{N^2}\sum_{i}
\sum_{j=-\frac{N}{2},j\ne i}^{\frac{N}{2}}  corr(x_i,x_{i+j})
\end{split}$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{1}{N^2}\sum_{i}
\sum_{j=-\frac{N}{2},j\ne i}^{\frac{N}{2}}  corr(x_i,x_{i+j}) \approx 
\frac{1}{N}
\sum_{j=-\frac{N}{2},j\ne i}^{\frac{N}{2}}  corr(x_0,x_{j})=2\frac{1}{N}
\sum_{j=1}^{\frac{N}{2}}  corr(x_0,x_{j})
$$

+++ {"slideshow": {"slide_type": "slide"}}

$$Var[\bar x] = \frac{Var[x]}{N}+ 2\frac{1}{N}
\sum_{j=1}^{\frac{N}{2}}  corr(x_0,x_{j}) = \frac{Var[x]}{N}\left(1+2\sum_{j=1}^{\frac{N}{2}}  \frac{corr(x_0,x_{j})}{Var[x]} \right)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Autocorrelation

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
lag, ac ,_,_= plt.acorr(chain[1000:100000]-chain[1000:100000].mean(),maxlags=100, usevlines=False, marker='.')
```

+++ {"slideshow": {"slide_type": "skip"}}

### Example - Student's t distribution

+++ {"slideshow": {"slide_type": "skip"}}

$$P(t) \propto \left(1+\frac{t^2}{n}\right)^{-\frac{n+1}{2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

$$\log P(t) = -\frac{n+1}{2}\log\left(1+\frac{t^2}{n}\right) + C$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def log_t(t,n=20):
    -0.5*(n+1.0)*no.log(1+t*t/n)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Example - Normal data with uninformative priors

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu,\sigma^2|y) \propto  (\sigma^2)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2},\quad s^2=\frac{n}{n-1}\left(\overline{y^2} -{\bar y }^2\right)
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def log_mu_sig2(mu, sig2, n, y_bar, s2):
    log_p = -0.5*(n+2)*np.log(sig2)
    log_p -= 0.5*n*(mu-y_bar)*(mu-y_bar)/(sig2)
    log_p -= 0.5*(n-1)*s2/sig2
    return log_p
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def mcmc2(log_p, gen, size, x):
   
    prev_x = x;
    prev_log_p =  log_p(*prev_x)
    chain = []
    chain.append(np.append(x,prev_log_p))
    for i in range (size):
        trial_x = gen(prev_x)
        trial_log_p = log_p(*trial_x)
        accept = True
        if(trial_log_p < prev_log_p):
            r = np.random.uniform(0,1)
            if(r>np.exp(trial_log_p-prev_log_p)):
                accept = False
        if accept:
            prev_x= trial_x
            prev_log_p = trial_log_p
        save = np.append(prev_x, prev_log_p)    
        chain.append(save)
    return np.asarray(chain)       
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
mu=1
sigma=2
y = np.random.normal(mu, sigma,20)
y_bar = y.mean()
s2 = y.var(ddof=1)
print(y_bar, s2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
log_p = lambda m , s: log_mu_sig2(m,s,20,y_bar, s2)
def make_gen(eps):
    def gen(x):
        trial = x + np.random.uniform(-1,1, 2)*eps
        trial[1] = np.abs(trial[1])
        return trial
    return gen    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
make_gen((0.5, 0.25))((0,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
chain = mcmc2(log_p,make_gen([1, 1]), 100000,[0,1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain[:1000,1])
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
lag, ac ,_,_= plt.acorr(chain[1000:100000,1]-chain[1000:100000,1].mean(),maxlags=100, usevlines=False, marker='.')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
lag, ac ,_,_= plt.acorr(np.random.uniform(-1,1,100000),maxlags=100, usevlines=False, marker='.')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain[1000:,0],bins=50, histtype='step', density=True)
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain[1000:,1],bins=50, histtype='step', density=True)
plt.show()
```

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sigma^2|y \sim \operatorname{Scale-Inv-}\chi^2(n-1,s^2)=  \operatorname{Scaled-Inv-Gamma}(\frac{n-1}{2},\frac{(n-1)s^2}{2})$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain[1000:,1],bins=50, histtype='step', density=True)
xs = np.linspace(1,15,100)
plt.plot(xs, st.invgamma(19/2.,scale 
                         = 19*s2/2.).pdf(xs))
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Example - Signal with background

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import sys
sys.path.append('../../src/')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import bda.signal_background as sb
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sig_back = sb.Signalbackground(1,2,0,2.12)
sig_back.x = np.arange(-6,7)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
sig_back.x
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
n0 = 30
sig_back.gen_counts(n0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(sig_back.rates,drawstyle='steps-mid')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(sig_back.rates,drawstyle='steps-mid')
plt.plot(sig_back.counts,'o')
```

+++ {"slideshow": {"slide_type": "slide"}}

```
def rate(x,A,B,mu, sigma):
    res = np.subtract.outer(mu,x)
    res2 = -0.5*res*res
    exponent = _swapaxes(np.multiply.outer(1.0/(sigma*sigma), res2),0,1)
    signal = np.multiply.outer(A,np.exp(exponent))
    output = np.add.outer(B,signal)
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.swapaxes(output,0,1)
    else:
        return output

def log_prob(counts, rates):
    return np.sum(counts*np.log(rates) - rates-sp.special.loggamma(counts), axis = -1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def log_p_sb(A,B):
    return sig_back.log_prob(A,B,0,2.12)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def make_gen(eps):
    def gen(x):
        trial = x + np.random.uniform(-1,1, 2)*eps
        trial=np.abs(trial)
        return trial
    return gen   
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
%%time 
chain = mcmc2(log_p_sb,make_gen((0.5, 0.5)), 400000,(1,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain[1000:,0], bins=50, density=True, histtype='step')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain[1000:,1], bins=50, density=True, histtype='step')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
As=np.linspace(0,1.5,100)
Bs=np.linspace(1.5,2.7,100)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
lp = sig_back.log_prob(As,Bs,0,2.12)
lp = lp-lp.max()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.contour(Bs, As, lp, levels=np.log([0.01, 0.1, 0.25,0.5, 0.75, 0.9, 0.95,1]))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(chain[1000:,1], chain[1000:,0])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(chain[1000:,1], chain[1000:,0])
plt.contour(Bs, As, lp, levels=np.log([0.01, 0.1, 0.25,0.5, 0.75, 0.9, 0.95,1]))
```

+++ {"slideshow": {"slide_type": "slide"}}

### Unknow $\mu$ and $\sigma$

```{code-cell} ipython3
def log_p_sb4(A,B,mu,sigma):
    return sig_back.log_prob(A,B,mu,sigma)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def make_gen4(eps):
    def gen(x):
        trial = x + np.random.uniform(-1,1, 4)*eps
        trial[0]=np.abs(trial[0]);
        trial[1]=np.abs(trial[1]);
        trial[3]=np.abs(trial[3]);
        return trial
    return gen   
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time 
chain4 = mcmc2(log_p_sb4,make_gen4((0.25, 0.25,0.25,0.25)), 1000000,(1,1,1,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
chain4.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain4[1000:,0])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain4[1000:,0], bins=50, density=True, histtype='step')
plt.grid()
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain4[:,1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain4[:,1], bins=50, density=True, histtype='step')
plt.grid()
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain4[:,2])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain4[1000:,2], bins=50, density=True, histtype='step')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain4[:,3])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain4[:,3], bins=100, density=True, histtype='step')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(chain4[1000:,1], chain4[1000:,0])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
plt.plot(chain4[1000:,0], chain4[1000:,2],'.')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain4[1000:,4])
```

+++ {"slideshow": {"slide_type": "slide"}}

#  ???

+++ {"slideshow": {"slide_type": "slide"}}

### Unknow $\mu$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def log_p_sb3(A,B,mu):
    return sig_back.log_prob(A,B,mu,2.12)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def make_gen3(eps):
    def gen(x):
        trial = x + np.random.uniform(-1,1, 3)*eps
        trial[0]=np.abs(trial[0]);
        trial[1]=np.abs(trial[1]);
        return trial
    return gen   
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
%%time 
chain3 = mcmc2(log_p_sb3,make_gen3((0.25, 0.25,0.25)), 1000000,(1,1,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
chain3.shape
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(chain3[:,2])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(chain3[1000:,2], bins=50, density=True)
plt.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

### PyMC3

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import pymc3 as pm
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sb_model2 = pm.Model()

with sb_model2:
    A = pm.HalfFlat('A')
    B = pm.HalfFlat('B')
    mu = 0.0
    sigma = 2.12
    rates = n0*( A*pm.math.exp(-0.5*(sig_back.x-mu)*(sig_back.x-mu)/(sigma*sigma))+B)
    count_Obs = pm.Poisson('count_Obs', mu=rates, observed=sig_back.counts)
    #pm.find_MAP(model=sb_model)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
pm.find_MAP(model=sb_model2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with sb_model2:
    trace2 = pm.sample(draws=200000, cores=4, step=pm.Metropolis())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
_ = pm.traceplot(trace2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
sb_model4 = pm.Model()

with sb_model4:
    A = pm.HalfNormal('A', sd=10)
    B = pm.HalfFlat('B')
    mu = pm.Normal('mu', mu=0, sd=3)
    sigma =pm.Gamma('sigma',mu=2, sd=2)
    rates = n0*( A*pm.math.exp(-0.5*(sig_back.x-mu)*(sig_back.x-mu)/(sigma*sigma))+B)
    count_Obs = pm.Poisson('count_Obs', mu=rates, observed=sig_back.counts)
    #pm.find_MAP(model=sb_model)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
sig_back.x
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
pm.find_MAP(model=sb_model4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---

```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with sb_model4:
    trace4 = pm.sample(draws=200000, cores=4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
_ = pm.traceplot(trace4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
trace4.get_values('A').min()
```
