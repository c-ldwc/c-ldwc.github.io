---
layout: post
title:  "Hamiltonian Monte Carlo"
date:   2024-11-16 21:46:32 +1100
categories: jekyll update
layout: single
classes: wide
excerpt: "Hamiltonian Monte Carlo From Scratch in Python"
use_math: true
---
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
  </script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
 $\renewcommand{\hat}[1]{\widehat{#1}}$

```python
# %conda install tqdm==4.66.5 scipy==1.13.1 plotly==5.24.1 numpy==1.26.4 pandas==2.2.2 openblas==0.3.21 matplotlib
```


```python
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import norm, multivariate_normal, uniform, beta

from samplers import hmc, metro_hastings, R_hat, plot_chains, param_scatter

n_iters = 3000
warmup = 1000
p = 3
n_param = p + 1
```

Markov Chain Monte Carlo (MCMC) algorithms draw samples from target probability distributions. The resulting samples can be used to approximate integrals (i.e. expectations) over the distribution being targeted. They are the workhorse of Bayesian computation, where posteriors are often too complex to solve without algorithmic tools, and are responsible for computation in the popular Bayesian software packages Stan and PyMC. 

I use PyMC fairly regularly, but my understanding of the MCMC algo it uses (a variant of HMC) was largely based on intuition. The analyses in the notebooks in this folder (`Portfolio/MCMC/`) are part of a self learning exercise where I use an implementation of Hamiltonian Monte Carlo (HMC) that I built in order to better understand how this class of algorithms works.

The HMC algorithm is based on the exposition in [Gelman et al.](https://stat.columbia.edu/~gelman/book/) chapters 10, 11, and 12. The MCMC functions live in `samplers/samplers.py`. There are some MCMC diagnostics in `samplers/utils.py`

The `hmc` function is a Hamiltonian Monte Carlo (HMC) sampler.  It requires 
- `log_prob`: an unnormed log probability for the distribution of interest, 
- `data`: a dictionary of data and parameters to pass to that log_prob. This is evaluated as part of each iteration of the algorithm
- `grad`: the gradient of the distribution with regard to the parameters and a starting point for the parameter samples. 
- `n_iters`: the number of iterations for the algorithm
- `starting`: a starting point for the samples
- `eps`, `L`, `M`: Tuning parameters for the algorithms Hamiltonian dynamics. See [Gelman et al.](https://stat.columbia.edu/~gelman/book/) chapter 12.


The algorithm generates a "momentum" variable at the start of each iteration. It uses this and the gradient to explore the target log density through a discrete approximation to hamiltonian dynamics in physics. At the end of an iteration, it computes a ratio $r$ of the target density at the starting values and the final location of the iteration. It accepts the new location as a sample with probability $\min(r,1)$ - this is the same as the Metropolis Hastings algorithm's acceptance step. This means that normalisation constants cancel and we can work with unnormalised log densities as our target algorithm. 

At the end of the interations, we have a "chain" of samples. A properly-tuned HMC run will converge to the target density, but that convergence takes time, so we drop a fixed number of starting iterations as a "warmup". Convergence to the target density can be measured by visual and diagnostic tests that I discuss later.

## Sampling From a Beta Distribution

To see how it works, imagine we want to use it to draw samples from a Beta(x\|a = 3,b = 5) distribution. We would never use MCMC for this in practice, because simpler methods like the ratio of gamma variables exist for Beta sampling, but it is a nice example of how these algorithms work.

The algorithm works best if its sample space is unbounded so the search doesn't reach areas of zero density. I use the logistic function to achieve this by mapping the real line to the support of the Beta distribution. The details of the derivation are in a footnote [^1] 


```python
def inv_logit(x): return 1/(1+np.exp(-x))

def log_prob_beta(proposal, a, b):
    x = inv_logit(proposal)
    return (a-1)*np.log(x) + (b - 1)*np.log(1-x)-proposal-2*np.log(1+np.exp(-proposal))

def grad_beta(proposal, a, b):
    inv_lgt_grad = 1/(1+np.exp(-proposal))**2 * np.exp(-proposal)
    x = inv_logit(proposal)
    return ((a-1)/x - (b-1)/(1-x)) * inv_lgt_grad - 1 + 2*np.exp(-proposal)/(1+np.exp(-proposal))

samples = hmc(
        M=1.,
        data={"a": 5, "b": 3},
        grad=grad_beta,
        n_iter=n_iters,
        log_prob=log_prob_beta,
        starting=[0],
        eps=.01,
        L=100,
    )
```

    accept rate: 1.0: 100%|████████████████████| 2999/2999 [00:13<00:00, 218.70it/s]


This results in the following chain of samples. The top plot is the untransformed variable. The lower plot is the inverse logit transformed variable. We can see that it's mapped to $[0,1]$ and concentrating around the top part of the interval


```python
fig, ax = plt.subplots(2, 1)
ax[0].plot(range(warmup, n_iters), samples[warmup:])
ax[0].set_title('Samples')
ax[1].plot(range(warmup, n_iters), inv_logit(samples[warmup:]))
ax[1].set_title('Transformed Samples')
fig.tight_layout();
```


    
![png](/assets/images/output_6_0.png)
    


The sample mean is close to the true mean of 0.625. The quantiles and histogram of the samples are also in accordance with the expected shape.


```python
print(f"sample mean {inv_logit(samples[warmup:]).mean()}, true mean {5/8}")
print(f"true .25, .5 and .75 quantiles {np.round((beta(5,3).ppf(.25),beta(5,3).ppf(.5), beta(5,3).ppf(.75)),3)}")
print(f"sample .25, .5 and .75 quantiles {np.round(np.quantile(inv_logit(samples[warmup:]), (.25, .5, .75)),3)}")
```

    sample mean 0.6268913144573977, true mean 0.625
    true .25, .5 and .75 quantiles [0.514 0.636 0.747]
    sample .25, .5 and .75 quantiles [0.51  0.645 0.744]



```python
x = np.linspace(0,1,100)

fig, ax = plt.subplots(2, 1)

ax[0].plot(x, beta(5,3).pdf(x), c='blue')

ax[1].hist(inv_logit(samples),50)

fig.tight_layout();
```


    
![png](/assets/images/output_9_0.png)
    


## Linear Regression

This is a model for a linear regression with gaussian errors, an inverse gamma prior $\mathrm{inv\\_\Gamma}(1.5, 1)$ on the standard deviation of the errors and independent normal priors on the coefficients of the model.

Using $X_{i\cdot}$ as the $i^{th}$ row of X, the matrix of predictors, and $\mathbf{y}$ as the vector of outcome observations. The posterior density is
$$p(\theta|\mathbf{y}) \propto \prod_{i=1}^{N}\frac{1}{\sigma^2} \exp \left(\frac{-(y_i - X_{i\cdot}\beta)^2}{2\sigma^2}\right) \frac{1}{\sigma^5}\exp\left({\frac{-1}{\sigma^2}}\right)\exp\left(\frac{-\beta^T\beta}{2}\right)$$

The gradient for the _log_ posterior for the coefficients is 

$$\sum_{i=1}^{N}\left[\frac{\left(y_i - X_{i\cdot}\beta\right)}{\sigma^2} X_{i\cdot}\right] - \beta^T$$

The partial derivative for sigma is 

$$ -5 - \frac{2(N-1)}{\sigma^2} + \sum_{i=1}^{N}\frac{y_i - X_{i\cdot}\beta}{\sigma^3} $$


```python
def log_prob(data, X, proposal):
    mu = X @ proposal[:-1]
    sigma = proposal[-1]  # variance

    log_lik = np.sum(-((data - mu) ** 2) / (2 * sigma**2)) - data.shape[0] * np.log(sigma**2)

    prior_coef = -np.dot(proposal.T, proposal) / 2
    prior_sigma = -2.5 * np.log(sigma**2) - 1 / sigma**2

    return log_lik + prior_coef + prior_sigma

def grad(data, X, proposal):
    mu = X @ proposal[:-1, :]
    sigma = proposal[-1, :]
    N = data.shape[0]

    coef_grad = (
        np.sum((data - mu) / (sigma**2) * X, axis=0).reshape(-1, 1) - proposal[:-1]
    )

    sigma_grad = [
        -5. - 2*(N-1)/(sigma**2) + np.sum((data - mu) ** 2 / (sigma**3))
    ]
    # print(sigma_grad)
    return np.r_[coef_grad, sigma_grad]


def create_regression(N=1000, p=2, sigma = 2):
    coef = uniform(-2,4).rvs(p)

    X = multivariate_normal(np.zeros(p), np.eye(p, p)).rvs(N)

    y = X @ coef + norm(0, sigma).rvs(N)
    y = y.reshape(-1, 1)
    coef = coef.reshape(-1, 1)
    return X, y, coef, sigma


X, y, coef, sigma = create_regression(N=1000, p=p)
```


```python
print(f"coefficients = {coef.flatten()}, sigma = {sigma}")
```

    coefficients = [ 0.72436258 -0.73060936 -1.74381374], sigma = 2


### Sampling the posterior and checking convergence

To check that our setup converges to the desired distribution, we run the sampler 4 times. Each sampler "chain" starts from different - ideally well separated - starting points. This is tricky to set up because if our starting points are too diffuse, the algorithm will take a long time to converge. If our starting points are all together, we run the risk of getting stuck in a local maximum. To check that we have converged, we want our runs to end up in the same region of posterior space. We can check this by visual inspection and the calculation of the $\hat{R}$ value, which is a ratio of between chain and within chain variance that will converge to one if the chains have mixed in the same region of posterior space.


```python
chains = np.ones([4, n_iters, coef.shape[0] + 1])

for chain in range(chains.shape[0]):
    params = hmc(
        M=np.eye(n_param, n_param) * 1.2,
        data={"data": y, "X": X},
        grad=grad,
        n_iter=n_iters,
        log_prob=log_prob,
        starting=np.r_[np.random.choice(y.flatten(), size = p, replace = False), np.exp(norm(0, .5).rvs(1))],
        eps=0.01,
        L=100,
    )
    chains[chain, :, :] = params
```

    accept rate: 0.998: 100%|██████████████████| 2999/2999 [00:25<00:00, 117.40it/s]
    accept rate: 1.0: 100%|████████████████████| 2999/2999 [00:25<00:00, 116.75it/s]
    accept rate: 1.0: 100%|████████████████████| 2999/2999 [00:25<00:00, 117.08it/s]
    accept rate: 1.0: 100%|████████████████████| 2999/2999 [00:25<00:00, 116.92it/s]


The $\hat{R}$ diagnostic value (the ratio of between and within chain variance) is less than 1.1 for all variables, and the chains appear to cover the same area in the plots, indicating that we have converged to the target distribution.


```python
R_hat(chains, warmup).round(3)
```




    array([1.   , 1.   , 1.001, 1.   ])




```python
plot_chains(chains, warmup, names = ['coef1', 'coef2', 'coef3', 'sigma'])
```


    
![png](/assets/images/output_17_0.png)
    



```python
coef
```




    array([[ 0.72436258],
           [-0.73060936],
           [-1.74381374]])




```python
samples = chains[:, warmup:, :].reshape(4 * (n_iters - warmup), n_param)

post_mu = X @ samples[:, :-1].T
```


```python
param_scatter(samples, warmup=0, names=["coef1", "coef2", "coef3", "sigma"], plot_params = {'alpha':.4, 's':4, 'c':'k'})
```


    
![png](/assets/images/output_20_0.png)
    



```python
coef
```




    array([[ 0.72436258],
           [-0.73060936],
           [-1.74381374]])




```python
samples.mean(axis=0)
```




    array([ 0.76133636, -0.64949313, -1.6823038 ,  2.06149481])




```python
np.quantile(samples, (0.025, 0.975), axis=0)
```




    array([[ 0.63253292, -0.77928289, -1.80824712,  1.93608431],
           [ 0.88583829, -0.5211314 , -1.55216088,  2.20137753]])




```python

```


```python
# post_preds = norm(post_mu, samples[:, -1]).rvs((1000, 4 * (n_iters - warmup)))

# post_pred_errors = post_preds - y.reshape(-1, 1)

# rand_obs = np.random.choice(range(y.shape[0]), 6, replace=False)

# fig = make_subplots(3, 2)

# for i, obs in enumerate(rand_obs):
#     # print(f"i {i} row {(i)//2 + 1} , col {(i) % 2 }")
#     fig.add_trace(
#         go.Histogram(x=post_pred_errors[obs]), row=(i) // 2 + 1, col=(i) % 2 + 1
#     )

#     fig.add_vline(x=0, row=(i) // 2 + 1, col=(i) % 2 + 1)
#     print(np.mean(post_pred_errors[obs] > 0))

# fig.show()
```


```python
# A = 1 / 25 * X.T @ X + np.eye(coef.shape[0], coef.shape[0])
# A_inv = np.linalg.inv(A)

# w = 1 / 25 * A_inv @ X.T @ y.flatten()
# post = multivariate_normal(w, A_inv)

# an_pts = 150

# x_plot = np.linspace(
#     params[warmup:, 0].min() - 0.05, params[warmup:, 0].max() + 0.05, n_pts
# )
# y_plot = np.linspace(
#     params[warmup:, 1].min() - 0.05, params[warmup:, 1].max() + 0.05, n_pts
# )

# z = np.ones((n_pts, n_pts))

# for i in tqdm(range(n_pts)):
#     for j in range(n_pts):
#         z[i, j] = post.pdf([x_plot[j], y_plot[i]])
```


```python
# A_inv
```


```python
# fig = go.Figure(go.Contour(x=x_plot, y=y_plot, z=z, ncontours=50))

# fig.add_scatter(x=params[warmup:, 0], y=params[warmup:, 1], opacity=0.4, mode="markers")
```


```python

```

[^1]: The density for the logit $y$ of the Beta variable is 

    $$f(y|a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}l(y)^{a-1}(1-l(y))^{b-1}l(y)^2\exp(-y)$$

    Where $l$ is the inverse logit function

    $$l(x) =  \frac{1}{1+\exp(-x)} $$

    The distribution's normalisation constant by definition doesn't depend on $x$ and we can ignore it for the purposes of the algorithm, so we use the unnormalised log density for the Beta distribution

    $$ f(y, a, b) = (a-1) \dot \log\left[l(y)\right] + (b-1) \dot \log\left[1-l(y)\right] - y - 2\log\left[1+\exp(-y)\right]$$




    so $l(x) \in [0,1]$.

    We need the derivative for this function in order to use HMC. It is 

    $$\frac{\partial f(y|a,b)}{\partial y} = \left(\frac{a-1}{l(y)} - \frac{b-1}{1-l(y)}\right) l(y)^2 \exp(-y) - 1 + 2l(y)\exp(-y)$$


```python

```
