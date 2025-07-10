---
layout: post
title:  "Fast Generalized Linear Models in Rust"
date:   2025-07-05
categories: Statistics Rust
layout: single
classes: wide
excerpt: "Building a GLM library from scratch in Rust with nalgebra"
use_math: true
toc: true
toc_label: "Table of Contents"
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
# Building Fast GLMs in Rust

Generalized Linear Models (GLMs) extend ordinary linear regression to handle non-normal response distributions and are a core part of applied statistics that every data scientist uses. While Python and R have mature statistical libraries, Rust's ecosystem for statistical computing is still developing. This post walks through a from-scratch implementation of GLMs in Rust that achieves ~300ms fits on a large (2.5M cell design matrix) dataset.

The complete implementation supports Poisson and Binomial families with Newton-Raphson optimization for parameter estimation. All code is available in the [`rust_glm`](https://github.com/c-ldwc/rust_glm) repository.

## Core Architecture

The library centers around a [Family trait](https://github.com/c-ldwc/rust_glm/blob/main/src/families/family.rs) that describes the necessary functions for the exponential family members and separates it from the optimization machinery:

```rust
pub trait Family {
    type Parameters;
    type Data;
    
    fn link(&self, mu: &DVector<f64>) -> DVector<f64>;
    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64>;
    fn log_lik(&self, mu: &DVector<f64>) -> f64;
    //Plus functions for various derivatives as well as the hessian and gradient
}
```

The `GLM` struct ingests a struct that has the family trait and uses this to inform the optimisation:

```rust
pub struct GLM<F: Family> {
    pub family: F,
    pub Data: DMatrix<f64>,     // Design matrix
    pub coef: DVector<f64>,     // Parameter estimates
    pub y: DVector<f64>,        // Response vector
    p: usize,                   // Number of parameters
}
```

## Exponential Family Mathematics

GLMs assume the response follows an exponential family distribution with density:

$$f(y|\theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)$$

Where $\theta$ is the natural parameter, $b(\theta)$ is the log partition function, and $a(\phi)$ controls dispersion.

The key insight is that $b'(\theta) = E[Y]$, which connects the natural parameter to the mean through the link function $g(\mu) = \theta$.

### Gradients and Hessians

The log-likelihood gradient has the general form:

$$\frac{\partial \ell}{\partial \beta} = X^T W G (y - \mu)$$

Where $\mu = g^{-1}(X\beta)$ and the matrices are defined as:

- $\alpha(\mu_i) = 1 + (y_i - \mu_i)\left\{\frac{V'(\mu_i)}{V(\mu_i)} + \frac{g''(\mu_i)}{g'(\mu_i)}\right\}$ where $g$ is the link function
- $G = \text{diag}\left\{\frac{g'(\mu_i)}{\alpha(\mu_i)}\right\}$ contains scaled link derivatives
- $W = \text{diag}(w_i)$ where $w_i = \frac{\alpha(\mu_i)}{g'(\mu_i)^2 V(\mu_i)}$ are the IRLS weights

The Hessian (Fisher information) is:

$$H = -X^T W X / \phi$$

Both the [gradient](https://github.com/c-ldwc/rust_glm/blob/67a6d065efa358f486bd7722c1528346913a8670/src/families/family.rs#L36) and [hessian](https://github.com/c-ldwc/rust_glm/blob/67a6d065efa358f486bd7722c1528346913a8670/src/families/family.rs#L20) functions avoid large matrix multiplications by scaling the relevant parts of the leftmost matrix in the appropriate definition:

```rust
fn hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
    // Compute -X^T * W * X efficiently
    let mut neg_X_t = self.get_data()
    .clone()
    .scale(-1.0)
    .transpose();
    let w = self.w(&x);

    // Scale columns of -X^T by weights
    for i in 0..neg_X_t.shape().1 {
        neg_X_t.column_mut(i).scale_mut(w[i]);
    }
    neg_X_t * self.get_data().scale(1.0/self.scale())
}
```
Because the hessian and gradient are calculated using the core exponential family functions, [both are part of the trait](https://github.com/c-ldwc/rust_glm/blob/67a6d065efa358f486bd7722c1528346913a8670/src/families/family.rs#L20) and need not be implemented directly. I think there is probably room for speed optimisations here with specific gradient or hessian functions for particular families

## Poisson Regression

For count data, we use the Poisson family with log link:

$$\log(\mu_i) = X_i \beta$$

The [Poisson implementation](https://github.com/c-ldwc/rust_glm/blob/main/src/families/poisson.rs) computes the log-likelihood directly:

```rust
impl Family for Poisson {
    fn link(&self, mu: &DVector<f64>) -> DVector<f64> {
        mu.map(|m| m.ln())
    }
    
    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64> {
        l.map(|L| L.exp())
    }

    fn V(&self, mu: &DVector<f64>) -> DVector<f64> {
        mu.clone()  // Variance equals mean for Poisson
    }

    fn log_lik(&self, mu: &DVector<f64>) -> f64 {
        let y = &self.y;
        let theta = mu.map(|m| m.ln());
        let b_theta = mu;
        
        // Handle log factorial efficiently
        let log_y_fact = y.iter().map(|&yi| {
            if yi <= 1.0 { 0.0 } else {
                (1..=(yi as u64)).map(|v| (v as f64).ln()).sum()
            }
        });

        y.iter().zip(theta.iter()).zip(b_theta.iter()).zip(log_y_fact)
            .map(|(((yi, thetai), bthetai), logyfacti)| {
                yi * thetai - bthetai - logyfacti
            })
            .sum()
    }

    //Plus functions for weights, alpha, link derivatives etc. 
}
```

## Binomial Regression

For binary or bounded count responses, the [implementation of the binomial distribution](https://github.com/c-ldwc/rust_glm/blob/main/src/families/binomial.rs) with parameters $n$ (trials) and $p$ (probability) uses the logit link. The mean parameter $\mu = np$ represents the expected number of successes:

$$\text{logit}(\mu/n) = \log\left(\frac{\mu}{n-\mu}\right) = X_i \beta$$

The implementation uses the rug crate for exact binomial coefficient computation:

```rust
impl Family for Binomial {
    fn link(&self, mu: &DVector<f64>) -> DVector<f64> {
        mu.map(|mu| (mu / (self.n - mu)).ln())
    }

    fn inv_link(&self, l: &DVector<f64>) -> DVector<f64> {
        l.map(|x| self.n / (1.0 + (-x).exp()))
    }

    fn V(&self, mu: &DVector<f64>) -> DVector<f64> {
        mu.map(|mu| mu * (1.0 - mu / self.n))
    }
    
    fn log_lik(&self, l: &DVector<f64>) -> f64 {
        let mu = self.inv_link(&l);
        let theta = self.link(&mu);
        let b = theta.map(|t| self.n * (1.0 + t.exp()).ln());
        
        // Binomial coefficient computation
        let n_int = Integer::from(self.n as i32);
        let c = self.y.map(|y| n_int.clone().binomial(y as u32).to_f64().ln());

        multizip((self.y.iter(), theta.iter(), b.iter(), c.iter()))
            .map(|(y, t, b, c)| (y * t - b) + c)
            .sum::<f64>()
    }

    //Plus functions for weights, alpha, link derivatives etc. 

}
```

## Newton-Raphson Optimization

The optimizer uses Newton's method with analytic gradients and Hessians:

```rust
pub fn optim(&mut self) -> Result<(bool), Box<dyn Error>> {
    let mut nab: DVector<f64> = self.family.grad(&self.coef);
    let mut hess: DMatrix<f64> = self.family.hessian(&self.coef);

    //Newton-Raphson with fixed step size
    for _i in 0..self.optim_args.max_iter {
        let lu = &hess.lu();
        let dir = lu.solve(&(-&nab)).ok_or("Hessian is not invertible")?;

        self.coef = &self.coef + dir;  // Take full Newton step

        nab = self.family.grad(&self.coef);

        // Check convergence
        if nab.norm().lt(&1e-5) {
            println!("Converged in {} iterations", _i);
            return Ok(true);
        }
        hess = self.family.hessian(&self.coef);
    }
    Ok(true)
}
```

The current implementation uses full Newton steps, but I plan to implement line search properly at some point.

## Statistical Inference

After optimization, we compute standard errors and confidence intervals using the Fisher information matrix:

```rust
fn inference(&self, alpha: f64) -> Result<inference_results, Box<dyn Error>> {
    // Fisher information matrix (inverse of negative Hessian)
    let covar = self
        .family
        .hessian(&self.coef)
        .scale(-1.0)
        .try_inverse()
        .ok_or("Hessian is not invertible")?;
    
    let q = Normal::standard().inverse_cdf(1.0 - alpha);  // Critical value
    let coef_var = covar.diagonal();  // Standard errors squared

    let mut p = vec![-1.0; self.coef.shape().0];
    let mut CI = vec![(0.0,0.0); self.coef.shape().0];

    // Calculate p-values and confidence intervals
    for i in 0..p.len() {
        let coef = self.coef[i];
        let n = Normal::new(0.0, coef_var[i].sqrt())?;
        p[i] = 2.0*(1.0-n.cdf(coef.abs()));  // Two-tailed test
        CI[i] = (coef - q * coef_var[i].sqrt(), coef + q * coef_var[i].sqrt());
    }

    Ok(inference_results{covar, p, CI})
}
```

## Usage Example

Here's how to fit a Poisson regression model:

```rust
use rust_glm::*;

fn main() -> Result<(), Box<dyn Error>> {
    // Generate simulated data
    let true_coef = DVector::from_vec(vec![0.2, 1.0, -0.34, 0.3, 0.0]);
    let simulated = poisson_simulate(100_000, true_coef.clone())?;
    
    // Create Poisson family
    let fam = Poisson::new(
        simulated.X.clone(),
        simulated.y.clone(),
        DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),  // Starting values
    );

    // Fit model
    let mut model = GLM::new(fam, simulated.X.clone(), simulated.y.clone());
    model.optim()?;

    // Display results
    println!("{}", model.summary(0.05)?);
    
    Ok(())
}
```

The summary output includes parameter estimates, p-values, and confidence intervals:

```
═══════════════════════════════════════
    Generalized Linear Model Summary
═══════════════════════════════════════
Observations:   100000
Parameters:         5
Scale:        1.0000
Log-lik:   -135247.8934

┌─────────────┬────────────┬──────────┬─────────────────────────┐
│  Parameter  │ Estimate   │ P-value  │     95% Conf. Int.      │
├─────────────┼────────────┼──────────┼─────────────────────────┤
│          β1 │   0.199891 │   0.0000 │ ( 0.198279,  0.201503)  │
│          β2 │   1.000241 │   0.0000 │ ( 0.998630,  1.001852)  │
│          β3 │  -0.339847 │   0.0000 │ (-0.341457, -0.338237)  │
│          β4 │   0.300154 │   0.0000 │ ( 0.298543,  0.301765)  │
│   Intercept │   0.000089 │   0.9287 │ (-0.001521,  0.001699)  │
└─────────────┴────────────┴──────────┴─────────────────────────┘
```

## Performance and Extensions
Currently, benchmarking a Binomial family optimisation with 500,000 observations and 5 data points with Criterion gives a mean time of 318.47 ms and a 95%CI of [315.99 ms, 321.08 ms]. This is pretty fast despite including the data simulation functios and without any serious performance optimisations (other than avoiding large diagonal matrices).

In the future, we could speed this up with

- Sparse matrix support for large datasets  
- Parallel computation of gradients
- More sophisticated line search methods
- Hessian approximation for non-positive-definite Hessians.
- Additional family distributions (Gamma, negative binomial, Cox, Tweedie)

The trait-based design makes adding new families straightforward by implementing the `Family` trait with the appropriate functions and log-likelihood.