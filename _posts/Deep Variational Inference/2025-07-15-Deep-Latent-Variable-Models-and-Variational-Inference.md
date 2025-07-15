---
layout: post
title: "Deep Latent Variable Models and Variational Inference"
date: 2025-07-15
categories: [machine-learning, variational-inference]
tags: [deep-learning, latent-variables, variational-inference, neural-networks]
math: true
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

$\newcommand{\posterior}{ p(z|x;\theta)}$
$\newcommand{\prior}{ p(z)}$
$\newcommand{\likelihood}{ p(x|z;\theta)}$
$\newcommand{\marglike}{ p(x;\theta)}$
$\newcommand{\variational}{ q(z|x;\phi)}$
$\newcommand{\KL}[2]{ \text{KL}(#1 \parallel #2)}$
$\newcommand{\ELBO}{ \mathcal{L}(\theta, \phi)}$
$\newcommand{\expectation}[2]{ \mathbb{E}_{#1}[#2]}$
$\newcommand{\Normal}[2]{ \mathcal{N}(#1, #2)}$
$\newcommand{\Real}{ \mathbb{R}}$
$\renewcommand{\argmax}{ \operatorname*{argmax}}$
$\renewcommand{\argmin}{ \operatorname*{argmin}}$

A latent variable model assumes that observed data $\mathrm{x} \in \Real^T$ has a likelihood that depends on a variable $z$, and some generative model parameters $\theta$, $p(\mathrm{x} \mid z ; \theta)$. $z$ itself has prior $p(z)$. We can make these models "deep" by implementing the probability $p(x_t \mid x; \theta)$ using a neural network. For instance, we may assume $z \sim \Normal{\mu}{I}$ for $z \in \Real^k$ and $p(x_t\mid z; \theta, W) = \mathrm{softmax}(Wh_t)$ where $h_t$ is the output of a recursive neural network $h_t = \mathrm{RNN}(x_{t-1}, h_{t-1},z, \theta)$ and $W$ and $\theta$ are learnable. Latent models allow us to: 
- Cluster documents by using a discrete latent space and calculating the posterior $\posterior$ for a given document. The cluster is $\underset{z}{\argmax}\ \posterior$ 
- Generate new values of x by sampling $z$. This, to me, feels like sampling the posterior predictive distribution. Although it is not strictly the same thing because we are not sampling from $\posterior$. Text generation is largely the domain of transformers at the moment, but we can apply this to other autoregressive time series (i.e. financial data) or stationary distributions. 
    - The resulting values of x can be used for anomaly detection, causal inference etc. 

I've been interested in deep learning for parameterisation of probability distributions for a while, because I constantly encounter situations where I want to flexibly describe how certain distributional parameters vary as functions of some data. I've tried using Gaussian Processes for this, but there are issues of size because the covariance matrix scales quadratically with the number of data points and needs to be inverted.

## Variational Inference

In some cases, these models can have tractable posteriors and we can find $\theta$ that maximises the marginal likelihood $\marglike$ via methods like Expectation Maximisation. Unfortunately, when we parameterise a latent variable model with a neural network, we lose the ability to write down the posterior and must resort to approximations.