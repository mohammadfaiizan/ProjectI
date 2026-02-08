# Fixed Income and Interest Rate Models

## Bond Pricing Fundamentals

### Zero-Coupon Bonds

A zero-coupon bond pays a single cash flow $F$ (face value, typically \$100) at maturity $T$. Its price is:

$$P(t,T) = Fe^{-r(T-t)}$$

where $r$ is the continuously compounded risk-free rate.

More generally, with time-varying rates:
$$P(t,T) = F \exp\left(-\int_t^T r(s)ds\right)$$

### Coupon Bonds

A coupon bond pays periodic coupons $C$ and face value $F$ at maturity. The price is:

$$P = \sum_{i=1}^{n} Ce^{-r_i t_i} + Fe^{-r_n T}$$

where $t_i$ are coupon payment dates and $r_i$ are discount rates for each period.

### Yield to Maturity

The yield to maturity (YTM) $y$ is the constant rate that equates the bond price to its cash flows:

$$P = \sum_{i=1}^{n} Ce^{-yt_i} + Fe^{-yT}$$

For annual coupons:
$$P = \sum_{i=1}^{n} \frac{C}{(1+y)^i} + \frac{F}{(1+y)^n}$$

### Discount Factors

The discount factor $D(t,T)$ is the price today of \$1 received at time $T$:

$$D(t,T) = P(t,T)/F = e^{-\int_t^T r(s)ds}$$

Discount factors are fundamental building blocks for fixed income pricing.

## Yield Curve

### Zero-Coupon Curve

The zero-coupon curve (spot curve) gives yields for zero-coupon bonds of various maturities:

$$y(T) = -\frac{\ln D(0,T)}{T}$$

### Forward Rates

The forward rate $f(t,T_1,T_2)$ is the rate agreed today for borrowing from $T_1$ to $T_2$:

$$f(t,T_1,T_2) = \frac{\ln(D(t,T_1)/D(t,T_2))}{T_2 - T_1}$$

The instantaneous forward rate is:
$$f(t,T) = -\frac{\partial \ln D(t,T)}{\partial T} = r(T) + (T-t)\frac{\partial r(T)}{\partial T}$$

### Relationship Between Rates

$$D(0,T) = \exp\left(-\int_0^T f(0,s)ds\right)$$

The spot rate is the average of forward rates:
$$y(T) = \frac{1}{T}\int_0^T f(0,s)ds$$

## Duration and Convexity

### Macaulay Duration

Macaulay duration measures the weighted average time to receive cash flows:

$$D_{Mac} = \frac{\sum_{i=1}^{n} t_i C_i e^{-yt_i}}{P}$$

where $C_i$ are cash flows at times $t_i$.

### Modified Duration

Modified duration measures price sensitivity to yield changes:

$$D_{Mod} = -\frac{1}{P}\frac{\partial P}{\partial y} = \frac{D_{Mac}}{1+y}$$

For small yield changes:
$$\Delta P \approx -D_{Mod} \cdot P \cdot \Delta y$$

### Convexity

Convexity measures the second-order price sensitivity:

$$C = \frac{1}{P}\frac{\partial^2 P}{\partial y^2} = \frac{\sum_{i=1}^{n} t_i^2 C_i e^{-yt_i}}{P}$$

The price change including convexity:
$$\Delta P \approx -D_{Mod} \cdot P \cdot \Delta y + \frac{1}{2}C \cdot P \cdot (\Delta y)^2$$

Convexity is always positive for option-free bonds, providing a benefit when yields move.

### DV01

Dollar value of a basis point (DV01) measures the price change for a 1bp yield change:

$$\text{DV01} = -\frac{\partial P}{\partial y} \times 0.0001 = D_{Mod} \cdot P \cdot 0.0001$$

### Key Rate Durations

Key rate durations measure sensitivity to specific points on the yield curve:

$$\text{KR01}_i = -\frac{1}{P}\frac{\partial P}{\partial y_i}$$

where $y_i$ is the yield at key maturity $i$. This allows hedging against non-parallel curve shifts.

## Short-Rate Models

Short-rate models specify the dynamics of the instantaneous risk-free rate $r(t)$.

### Vasicek Model

The Vasicek model assumes mean-reverting dynamics:

$$dr(t) = \kappa(\theta - r(t))dt + \sigma dW(t)$$

where:
- $\kappa > 0$: speed of mean reversion
- $\theta$: long-term mean
- $\sigma$: volatility
- $W(t)$: Brownian motion

**Bond pricing:** The zero-coupon bond price is:

$$P(t,T) = A(t,T)e^{-B(t,T)r(t)}$$

where:
$$B(t,T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}$$

$$A(t,T) = \exp\left(\left(\theta - \frac{\sigma^2}{2\kappa^2}\right)(B(t,T) - (T-t)) - \frac{\sigma^2}{4\kappa}B(t,T)^2\right)$$

**Properties:**
- Rates can become negative
- Analytical bond prices
- Mean-reverting

### Cox-Ingersoll-Ross (CIR) Model

The CIR model prevents negative rates:

$$dr(t) = \kappa(\theta - r(t))dt + \sigma\sqrt{r(t)}dW(t)$$

The square root ensures $r(t) \geq 0$ if $2\kappa\theta \geq \sigma^2$ (Feller condition).

**Bond pricing:** Similar exponential-affine form with different $A(t,T)$ and $B(t,T)$ involving modified Bessel functions.

**Properties:**
- Non-negative rates
- Volatility proportional to $\sqrt{r}$
- More complex than Vasicek

### Hull-White Model

The Hull-White (extended Vasicek) model allows calibration to initial yield curve:

$$dr(t) = (\theta(t) - \kappa r(t))dt + \sigma dW(t)$$

The time-dependent function $\theta(t)$ is chosen to match the initial term structure.

**Bond pricing:**
$$P(t,T) = \frac{P^{market}(0,T)}{P^{market}(0,t)}\exp\left(A(t,T) - B(t,T)r(t)\right)$$

where $P^{market}$ is the market-observed bond price.

**Properties:**
- Fits initial yield curve exactly
- Mean-reverting
- Can produce negative rates

### Black-Derman-Toy (BDT) Model

BDT is a lognormal model in discrete time:

$$\Delta \ln r(t) = \left(\theta(t) - \frac{\sigma'(t)}{\sigma(t)}\ln r(t)\right)\Delta t + \sigma(t)\Delta W(t)$$

In the binomial tree implementation:
- Rates are lognormally distributed
- Volatility can be time-dependent
- Calibrated to market prices

**Properties:**
- Positive rates
- Fits initial term structure and volatility structure
- More complex calibration

## Heath-Jarrow-Morton (HJM) Framework

HJM models the entire forward rate curve directly.

### Forward Rate Dynamics

The instantaneous forward rate $f(t,T)$ follows:

$$df(t,T) = \alpha(t,T)dt + \sigma(t,T)dW(t)$$

where $\alpha(t,T)$ is the drift and $\sigma(t,T)$ is the volatility.

### No-Arbitrage Condition

HJM shows that to prevent arbitrage:

$$\alpha(t,T) = \sigma(t,T)\int_t^T \sigma(t,s)ds$$

The drift is determined by the volatility structure.

### Short Rate

The short rate is:
$$r(t) = f(t,t)$$

And:
$$r(t) = f(0,t) + \int_0^t \alpha(s,t)ds + \int_0^t \sigma(s,t)dW(s)$$

### Implementation

HJM is typically implemented via simulation of forward rates or through finite difference methods on the PDE.

## LIBOR Market Model (BGM)

The Brace-Gatarek-Musiela (BGM) model, also called the LIBOR Market Model, models discrete forward rates directly.

### Forward LIBOR Rates

The forward LIBOR rate for period $[T_i, T_{i+1}]$ is:

$$L_i(t) = \frac{P(t,T_i) - P(t,T_{i+1})}{\tau_i P(t,T_{i+1})}$$

where $\tau_i = T_{i+1} - T_i$ is the day count fraction.

### Dynamics

Under the $T_{i+1}$-forward measure:

$$dL_i(t) = \sigma_i(t)L_i(t)dW_i(t)$$

Under the spot measure (more realistic):

$$dL_i(t) = L_i(t)\sum_{j=\eta(t)}^{i}\frac{\tau_j L_j(t)\sigma_i(t)\sigma_j(t)\rho_{ij}}{1 + \tau_j L_j(t)}dt + \sigma_i(t)L_i(t)dW_i(t)$$

where $\eta(t)$ is the index of the next reset date.

### Caplet Pricing

A caplet is a call option on a LIBOR rate. Under the $T_{i+1}$-forward measure:

$$V_{caplet}(0) = \tau_i P(0,T_{i+1})\mathbb{E}^{T_{i+1}}[\max(L_i(T_i) - K, 0)]$$

If $\sigma_i(t)$ is deterministic, this gives Black's formula:

$$V_{caplet}(0) = \tau_i P(0,T_{i+1})[L_i(0)N(d_1) - KN(d_2)]$$

where:
$$d_1 = \frac{\ln(L_i(0)/K) + \sigma_i^2 T_i/2}{\sigma_i\sqrt{T_i}}$$
$$d_2 = d_1 - \sigma_i\sqrt{T_i}$$

and $\sigma_i^2 = \frac{1}{T_i}\int_0^{T_i}\sigma_i^2(s)ds$ is the average volatility.

### Swaption Pricing

Swaptions are options on swaps. Pricing requires simulation under the appropriate measure or approximations.

**Approximation:** Freeze the drift at today's forward rates, leading to approximate Black-Scholes formulas.

## Yield Curve Construction

### Bootstrapping

Bootstrapping constructs the zero-coupon curve from market instruments.

**Process:**
1. Start with shortest maturity instruments (deposits, futures)
2. Extract discount factors
3. Use longer instruments (swaps) to extend the curve
4. Solve for unknown discount factors

**Example:** From a swap with fixed rate $R$:

$$P(0,T_n) = \frac{1 - R\sum_{i=1}^{n-1}\tau_i P(0,T_i)}{1 + R\tau_n}$$

Solve recursively for $P(0,T_n)$.

### Interpolation Methods

**Linear interpolation:** Simple but can produce unrealistic forward rates

**Cubic spline:** Smooth curve, but may oscillate

**Nelson-Siegel:** Parametric form:
$$y(T) = \beta_0 + \beta_1\frac{1 - e^{-\lambda T}}{\lambda T} + \beta_2\left(\frac{1 - e^{-\lambda T}}{\lambda T} - e^{-\lambda T}\right)$$

**Svensson (Nelson-Siegel-Svensson):** Extended with additional term for more flexibility

### Smoothing

Yield curves should be smooth to avoid arbitrage. Forward rates must satisfy:
$$f(t,T) \geq -\frac{1}{T-t}$$

to prevent negative discount factors.

## Interest Rate Derivatives

### Interest Rate Swaps

A swap exchanges fixed rate payments for floating (LIBOR) payments.

**Fixed leg:** Pays $R_{fixed}$ periodically
**Floating leg:** Pays $L_i$ (LIBOR) periodically

**Swap rate:** The fixed rate making the swap value zero:
$$R_{swap} = \frac{\sum_{i=1}^{n}\tau_i P(0,T_i)L_i(0)}{\sum_{i=1}^{n}\tau_i P(0,T_i)}$$

### Caps and Floors

**Cap:** Portfolio of caplets, protects against rising rates
**Floor:** Portfolio of floorlets, protects against falling rates

Pricing: Sum of individual caplet/floorlet prices.

### Swaptions

**Payer swaption:** Right to enter a swap paying fixed
**Receiver swaption:** Right to enter a swap receiving fixed

Pricing requires modeling the swap rate dynamics, often using the LMM or approximations.

## Calibration

Interest rate models must be calibrated to market prices:

1. **Cap/floor volatilities:** Calibrate LMM to caplet implied volatilities
2. **Swaption volatilities:** Calibrate to swaption implied volatilities
3. **Yield curve:** Fit initial term structure
4. **Volatility surface:** Match implied volatilities across strikes and maturities

Calibration typically involves:
- Choosing volatility functions $\sigma_i(t)$
- Optimizing parameters to minimize pricing errors
- Regularization to ensure smooth and stable parameters
