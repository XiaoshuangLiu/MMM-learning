
# Prophet Decomposition & ALS-Based Baselining

This document provides an overview of two approaches to time-series or baseline decomposition often used in analytics:

1. **Prophet Decomposition**  
2. **ALS-based Baselining**

Below, you’ll find detailed explanations of the math behind each approach, how to interpret the resulting components, guidelines on when (and when not) to use them, and how controlling factors come into play.

---

## 1. Prophet Decomposition

### What is Prophet?

[Facebook (Meta) Prophet](https://github.com/facebook/prophet) is an open-source time series forecasting library. It models a time series \( y(t) \) as an **additive** combination of several components:

```text
y(t) = g(t) + s(t) + h(t) + ε(t),
```
where:
- g(t) is the trend component.
- s(t) is the seasonality component (potentially multiple seasonalities).
- h(t) is the holiday or special event component.
- ε(t) is the residual (noise) term.


### Prophet Decomposition Definition

**Prophet decomposition** refers to taking a fitted Prophet model and splitting (or “decomposing”) the observed time series values into each of these components. It answers the question: *How much of a given data point at time \( t \) is explained by the long-term trend? By seasonality? By special events? By random noise?*

### Math Behind Prophet’s Components

#### 1. Trend (\( g(t) \))

Prophet generally models the long-term trend using either a **piecewise linear** function or a **logistic** growth function (if there is a known carrying capacity). By default, Prophet uses the piecewise linear model.

##### 1.1 Piecewise Linear Trend

- **Core Idea**: We allow the slope of the trend to change at a set of predefined “change points.”
- **Mathematical Form** (simplified):
  \[
  g(t) = k + a(t) + \sum_{i=1}^{m} \kappa_i \cdot D_i(t),
  \]
  where:
  - \( k \) is the intercept at \( t=0 \).
  - \( a(t) \) is the overall linear slope up to time \( t \).
  - \( \kappa_i \) are the adjustments to the slope at each change point \( i \).
  - \( D_i(t) \) is an indicator function that is 1 when \( t \) is after the \( i \)-th change point and 0 otherwise.

The function \( g(t) \) therefore *segments* time into regions where the trend slope can differ, reflecting potential structural changes in the data over time.

##### 1.2 Logistic Growth (Alternative)

- **Core Idea**: If there is a **carrying capacity** \( C \) (for example, a saturation point in demand), Prophet can model:
  \[
  g(t) = C \Big/ \Bigl( 1 + \exp\bigl(-(\alpha + \beta t)\bigr) \Bigr),
  \]
  and may still include piecewise adjustments for changing growth rates.  
- This is helpful when there’s a known upper bound beyond which the metric cannot grow.

---

#### 2. Seasonality (\( s(t) \))

Prophet captures seasonality using **Fourier series**. If \( P \) is the period (e.g., \( P=7 \) for weekly seasonality if the data is daily, \( P=365.25 \) for yearly seasonality), then the seasonality function is approximated by a sum of sine and cosine terms.

\[
s(t) = \sum_{n=1}^{N} \Bigl( a_n \sin\bigl(\tfrac{2\pi n t}{P}\bigr) + b_n \cos\bigl(\tfrac{2\pi n t}{P}\bigr) \Bigr),
\]

where:
- \( N \) is the number of Fourier terms (the more terms, the more flexible the seasonality curve).
- \( a_n, b_n \) are learned coefficients during model fitting.

Prophet can handle multiple seasonalities by summing separate Fourier expansions, one for each seasonal cycle (e.g. daily, weekly, yearly).

---

#### 3. Holidays / Special Events (\( h(t) \))

To model the effect of holidays or special events, Prophet adds **additional regressors** that turn “on” around certain dates. Conceptually:

\[
h(t) = \sum_{k=1}^{K} \alpha_k \cdot X_k(t),
\]

where:
- \( X_k(t) \) is a binary indicator (or set of indicators) for holiday/event \( k \) at time \( t \).
- \( \alpha_k \) is the estimated impact of holiday \( k \).

This term allows the model to capture one-off spikes or drops in the data (e.g. Black Friday, Chinese New Year, promotions, etc.), which would not be fully explained by the standard seasonality or trend terms.

---

#### 4. Noise / Residual (\( \varepsilon(t) \))

The noise term \(\varepsilon(t)\) captures random fluctuations or **unexplained** variance in the data after accounting for the trend, seasonality, and holiday effects. In Prophet’s additive model, \(\varepsilon(t)\) is assumed to be independent noise. Typically, Prophet uses MAP (maximum a posteriori) or MCMC to estimate parameters, implicitly assuming these residuals are normally distributed or close to random white noise.

---

### Summary

Putting it all together, the Prophet model is:

\[
y(t) = \underbrace{g(t)}_{\text{trend}} \;+\; \underbrace{s(t)}_{\text{seasonality}} \;+\; \underbrace{h(t)}_{\text{holidays}} \;+\; \underbrace{\varepsilon(t)}_{\text{noise}}.
\]

1. **Trend**: Long-term patterns, which can be piecewise linear or logistic.  
2. **Seasonality**: Captured via Fourier series for daily, weekly, yearly, or other cycles.  
3. **Holidays**: Modeled as additional (mostly) short-term effects around specific dates or events.  
4. **Noise**: The leftover random variation.

By decomposing \( y(t) \) into these four parts, Prophet helps analysts understand and forecast the time series in a transparent, interpretable way.


1. **Trend (\( g(t) \))**  
   Often modeled as a **piecewise linear** function with potential change points (where the slope can change) or a logistic growth function if there is a known carrying capacity. For piecewise linear:

   ```text
   g(t) = k + a(t) + ∑(κᵢ · Dᵢ(t)),
   ```

   - \( k \) is the initial intercept.
   - \( a(t) \) captures the overall slope up to time \( t \).
   - \( κᵢ \) are the changes in slope, and \( Dᵢ(t) \) are indicators for whether time \( t \) is after change point \( i \).

2. **Seasonality (\( s(t) \))**  
   Modeled using a **Fourier series**:

   ```text
   s(t) = ∑( aₙ sin( (2πn t) / P ) + bₙ cos( (2πn t) / P ) ),
   ```
   where:
   - \( P \) is the period (e.g., 7 for weekly seasonality if daily data, 365.25 for yearly seasonality),
   - \( n \) is the order of the Fourier series,
   - \( aₙ \), \( bₙ \) are coefficients learned by the model.

   Prophet can handle multiple seasonalities simply by adding more Fourier expansions (e.g., one for weekly, one for yearly).

3. **Holiday Effects (\( h(t) \))**  
   Captures the influence of known special events or holidays (like Christmas, Black Friday, major promotions) as an additive term, often with shrinkage priors to prevent overfitting.

4. **Residual (\( ε(t) \))**  
   The unexplained noise once trend, seasonality, and holiday effects are taken into account.

### When to Use Prophet Decomposition

1. **Clear seasonality and trend.** If your data exhibits strong weekly, monthly, or yearly cycles plus overall growth or decline.
2. **Moderate-to-large historical data.** Prophet needs enough data (~1.5–2 cycles of the strongest seasonality) to robustly fit seasonal patterns.
3. **Known special events.** Prophet can incorporate specific holidays or promotions into the model as additional components.
4. **Ease of use.** Prophet is designed for relatively automated decomposition/forecasting without requiring deep time-series expertise.

### When *Not* to Use Prophet Decomposition

1. **Very short time series.** Prophet’s seasonal and trend estimations may be unreliable or overfit if you have fewer than ~1 full cycle of your major seasonality.
2. **Highly complex patterns.** If your data has major irregularities or structural breaks not well-captured by piecewise linear/logistic trends or standard Fourier-based seasonality.
3. **Specialized models needed.** Situations where ARIMA/state-space/HMM or hierarchical approaches fit domain needs better.
4. **Non-additive structures.** If the data relationships are largely multiplicative or dominated by large external shocks that Prophet can’t easily model.

### Major Prophet Components and How to Interpret Them

1. **Trend Plot**  
   - Depicts the overall level of the time series evolving over time, showing potential breakpoints.
   - **Reading**: If there’s a sustained slope upwards, it means the model sees consistent growth; changes in slope reflect potential shifts in growth rate.

2. **Seasonality Plots**  
   - Display weekly, yearly, or other seasonal cycles.
   - **Reading**: For weekly seasonality, you might see that Sundays have a higher contribution than Mondays, etc.

3. **Holiday Effects**  
   - Show the estimated lift (or dip) around specific dates or ranges.
   - **Reading**: A spike for something like Black Friday if your domain suggests higher activity on that date.

4. **Residual / Noise**  
   - The difference between the fitted values and actual data.
   - **Reading**: Ideally, no obvious remaining pattern implies the trend/seasonality/holidays captured most of the structure.

### What is a “Good” Decomposition?

- **Low residual variance** (i.e., the model fits data well without leftover structure).
- **Plausible component shapes** (the trend and seasonality make sense with domain knowledge).
- **Stable forecasting performance** (tested via cross-validation or holdout data).

---

## 2. ALS-Based Baselining

### What is ALS (Alternating Least Squares)?

In many advanced analytics contexts (e.g. **Marketing Mix Modeling**), **ALS-based baselining** is used to iteratively solve for a “baseline” component alongside multiple explanatory variables. A simplified example:

```text
y(t) = B(t) + ∑( βᵢ xᵢ(t) ) + ε(t),
```

where:

- \( B(t) \) is the baseline term.
- \( βᵢ \) capture the influence of controlling factor \( xᵢ(t) \).
- **ALS** iterates between solving for \( B(t) \) given fixed \( βᵢ \) and vice versa until convergence.

### When to Use ALS-Based Baselining

1. **Multiple known drivers.** You have many explanatory factors (e.g., marketing channels, competitor data, macro trends) that must be separated from a baseline.
2. **Need for iterative approach.** Standard regression might not suffice if you have constraints or non-linear transformations requiring iterative solves.
3. **Marketing Mix or advanced analytics.** Commonly used to parse out baseline from incremental lifts due to marketing or external interventions.

### When *Not* to Use ALS-Based Baselining

1. **Simplicity.** If you only have a single factor or purely time-series patterns with no additional drivers, a simpler method (e.g., Prophet, ARIMA) could suffice.
2. **No need for iteration.** If a basic regression or time-series model meets the objective.

### How Are Controlling Factors Captured? Why?

- **Controlling factors** (or control variables) are explicitly modeled as regressors:
  ```text
  y(t) = B(t) + ∑( fᵢ( xᵢ(t) ) ) + ε(t).
  ```
  - Each \( fᵢ(\cdot) \) can be linear or non-linear.
  - **Reason:** Without these factors, you might incorrectly attribute their effects to the baseline (or other components).

### Can We Read Out the Contribution of Controlling Factors?

Yes. Once fit, you have:

```text
ŷ(t) = B̂(t) + ∑( β̂ᵢ xᵢ(t) ).
```

where:
- \( B̂(t) \) is the estimated baseline at time \( t \).
- \( β̂ᵢ xᵢ(t) \) is the partial contribution of factor \( i \).

This allows you to see how each factor plus baseline sum to the total prediction.

---

## Putting It All Together

1. **Prophet Decomposition**  
   - Ideal for **time-series** with strong seasonality/trend + known holidays.
   - Additive structure: \( y(t) = \text{trend} + \text{seasonality} + \text{holiday} + \text{noise} \).
   - Out-of-the-box, user-friendly, interpretable.

2. **ALS-Based Baselining**  
   - Often used in **marketing mix modeling** or advanced scenarios with many variables.
   - Separates out a baseline from multiple controlling factors.
   - Iterative approach can handle constraints, non-linearities, or large sets of regressors.

Both methods **decompose** a time series into understandable components, but differ in how they handle external drivers and what they define as “baseline.” Choose the tool that best fits your data’s structure and your analytic goals.
```
