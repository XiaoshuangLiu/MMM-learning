# 📘 Full Explanation: Bayesian Regression, MCMC, and Trace (as in MMMResults)

---

## 1. Bayesian Regression Model

### What is it?
A **Bayesian regression model** estimates relationships between inputs (like media spend, promotions, seasonality) and an outcome (like sales), while quantifying uncertainty. Instead of giving one "best" value for each parameter, it gives a **distribution** of likely values based on both prior knowledge and observed data.

\\[
P(\\theta \\mid D) = \\frac{P(D \\mid \\theta) \\cdot P(\\theta)}{P(D)}
\\]

Where:
- \\( \\theta \\): model parameters (e.g., regression coefficients like \\(\\beta_1, \\beta_2\\))
- \\( D \\): observed data (e.g., sales, media spend)
- \\( P(\\theta) \\): **Prior** — your belief about the parameters before seeing the data
- \\( P(D \\mid \\theta) \\): **Likelihood** — how well the model explains the data, given \\(\\theta\\)
- \\( P(\\theta \\mid D) \\): **Posterior** — your updated belief after seeing the data
- \\( P(D) \\): **Evidence** (a normalizing constant, not needed in MCMC)

---

## 📘 What Each Term Means

| Term       | Description |
|------------|-------------|
| **Prior** \\( P(\\theta) \\)       | Encodes our beliefs about parameters before seeing any data (e.g., media ROAS is probably positive, decay is between 0 and 1) |
| **Likelihood** \\( P(D \\mid \\theta) \\) | Measures how probable the observed data is, assuming a certain value of the parameters |
| **Posterior** \\( P(\\theta \\mid D) \\)  | Combines prior beliefs and observed evidence — gives a full distribution of what we now believe about parameters |
| **Evidence** \\( P(D) \\)          | Just a normalizing constant so the posterior integrates to 1; not needed in MCMC |

---

```
P(θ | D) = [P(D | θ) * P(θ)] / P(D)
```
- Posterior = what we want (probability of parameters given data)
- Likelihood = how well the parameters explain the data
- Prior = our belief before seeing data
- Normalizing constant = total probability of the data

> We often write:  
> `Posterior ∝ Likelihood × Prior`  
> because the denominator is usually intractable.

### Intuition example:
Let’s say you observe someone coughing. You want to infer whether they have a cold.
- **Prior**: P(Cold)=0.1 → You believe there's a 10% chance someone has a cold.
- **Likelihood**: P(Cough｜Cold)=0.8 → 80% of people with a cold tend to cough.
- **Marginal probability of cough (normalizing constant/evidence/marginal likelihood)**:P(Cough)=0.2 → In the general population, 20% of people have a cough.
- 🧠 Question: What is the probability that someone has a cold given that they are coughing?
- **Posterior**: Combine the above to infer the most likely condition

### Common Distributions and Use Cases

| Distribution | Formula | Use Case |
|--------------|---------|----------|
| Normal (Gaussian) | N(μ, σ²) | Modeling noise, continuous parameters like ROAS |
| Half-Normal | HalfNormal(σ) | Non-negative parameters like decay |
| Uniform | Uniform(a, b) | Flat or bounded prior |
| Bernoulli | Bernoulli(p) | Binary success/failure |
| Binomial | Binomial(n, p) | Success count in n trials |
| Beta | Beta(α, β) | Modeling probabilities |
| Gamma | Gamma(α, β) | Skewed priors or waiting times |
| Exponential | Exp(λ) | Modeling time until event |
| LogNormal | LogNormal(μ, σ) | Positive-skewed variables |

---

## 2. Why MCMC?

### Problem:
We can’t compute the denominator of Bayes' Rule analytically. It involves integrating over all possible θ — which is hard in high dimensions or nonlinear models.

### Why is the numerator easy but denominator hard?
- **Numerator** = likelihood × prior → easy because it only needs evaluation at a single θ
- **Denominator** = integration over all θ → hard, especially with many dimensions

### Solution: MCMC
Use **random sampling** to approximate the posterior instead of calculating it.

### Markov Chain?
A sequence of values where the **next value only depends on the current value**, not the full history.

### Monte Carlo?
A method of using **randomness** to solve deterministic problems, like estimating areas or integrals.

---

## 3. MCMC Basics

### Metropolis-Hastings (MH)
1. Start with θ₀
2. Propose θ* (add random noise)
3. Compute acceptance ratio:
   ```
   r = P(D|θ*) * P(θ*) / P(D|θ₀) * P(θ₀)
   ```
4. Draw u ~ Uniform(0, 1)
5. Accept θ* if u < r
6. Else, stay at θ₀

✅ Even if θ* is worse (r < 1), it can still be accepted to escape local optima.

---

## 4. Hamiltonian Monte Carlo (HMC)

### Core idea:
Simulate physics — let θ be position and add artificial momentum p to guide the sampler through space.

### Components:
- θ = position (parameter)
- p = momentum (random)
- U(θ) = potential energy = -log(posterior)
- K(p) = kinetic energy
- H = total energy = U + K

### Leapfrog algorithm:
- Simulate a trajectory using gradients of the posterior
- Typically run 10–100 small steps
- Accept or reject the final position based on energy conservation

✅ Only the **last position** in the trajectory becomes the sample (others are internal to leapfrog).

---

## 5. How Bayesian Modeling + MCMC Work Together

| Step | Task |
|------|------|
| 1 | Define model: prior × likelihood |
| 2 | Posterior is intractable |
| 3 | MCMC samples from posterior |
| 4 | Result = a trace of θ samples |
| 5 | Analyze samples (mean, CI, uncertainty) |

---

## 6. What Are Traces? (as in `MMMResults`)

A **trace** is a collection of sampled values for all model parameters, generated by MCMC.

Each row is a sample of θ — one draw from the posterior.

### Common θ in MMM:
- Channel coefficients (ROAS)
- Adstock decay rates
- Baseline / Intercept
- Noise scale (σ)
- Seasonal controls

### Why do traces matter?
- Estimate mean, median, CI
- Visualize uncertainty
- Diagnose convergence
- Enable posterior plots and decisions

---

## ✅ Final Summary

- **Bayesian modeling defines** the structure of the problem
- **MCMC (especially HMC)** enables us to sample from complex posteriors
- **Trace = the core output** used to analyze media effect, ROAS, decay, uncertainty
