# Fourier Methods and FFT Pricing

## Introduction

Fourier transform methods provide efficient algorithms for option pricing, especially for models with known characteristic functions. The Fast Fourier Transform (FFT) enables rapid computation of prices for many strikes simultaneously. This document covers Fourier transforms, FFT algorithms, and pricing methods.

## Fourier Transform

### Definition

**Fourier transform:**
$$\hat{f}(u) = \int_{-\infty}^{\infty} f(x) e^{-iux} dx$$

**Inverse Fourier transform:**
$$f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(u) e^{iux} du$$

where $i = \sqrt{-1}$.

### Properties

**Linearity:**
$$\mathcal{F}[af + bg] = a\mathcal{F}[f] + b\mathcal{F}[g]$$

**Convolution:**
$$\mathcal{F}[f * g] = \mathcal{F}[f] \times \mathcal{F}[g]$$

where $(f * g)(x) = \int f(y)g(x-y)dy$.

**Derivative:**
$$\mathcal{F}[f'](u) = iu \mathcal{F}[f](u)$$

**Shift:**
$$\mathcal{F}[f(x-a)](u) = e^{-iua} \mathcal{F}[f](u)$$

## Characteristic Function

### Definition

For random variable $X$ with density $f(x)$:

**Characteristic function:**
$$\phi_X(u) = E[e^{iuX}] = \int_{-\infty}^{\infty} e^{iux} f(x) dx$$

**Properties:**
- $\phi_X(0) = 1$
- $|\phi_X(u)| \leq 1$
- $\phi_X(-u) = \overline{\phi_X(u)}$ (conjugate symmetry)

### Log-Price Characteristic Function

For stock price $S_T = S_0 e^{X_T}$ where $X_T = \ln(S_T/S_0)$:

**Characteristic function:**
$$\phi_{X_T}(u) = E[e^{iuX_T}]$$

**For geometric Brownian motion:**
$$X_T = (r - \frac{\sigma^2}{2})T + \sigma W_T$$

**Characteristic function:**
$$\phi_{X_T}(u) = \exp\left(iu(r - \frac{\sigma^2}{2})T - \frac{1}{2}u^2\sigma^2 T\right)$$

### Heston Model

**Characteristic function:**
$$\phi_{X_T}(u) = \exp(C(u,T) + D(u,T)V_0 + iuX_0)$$

where:
$$C(u,T) = r iu T + \frac{\kappa\theta}{\sigma_V^2}\left[(\kappa - \rho\sigma_V iu + d)T - 2\ln\left(\frac{1-ge^{-dT}}{1-g}\right)\right]$$

$$D(u,T) = \frac{\kappa - \rho\sigma_V iu + d}{\sigma_V^2} \frac{1-e^{-dT}}{1-ge^{-dT}}$$

with:
$$d = \sqrt{(\rho\sigma_V iu - \kappa)^2 + \sigma_V^2(iu + u^2)}$$
$$g = \frac{\kappa - \rho\sigma_V iu + d}{\kappa - \rho\sigma_V iu - d}$$

## FFT Algorithm

### Discrete Fourier Transform

**DFT:**
$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}$$

for $k = 0, 1, \ldots, N-1$.

**Inverse DFT:**
$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{2\pi i kn/N}$$

### Fast Fourier Transform

**Cooley-Tukey algorithm:**

Recursively divide:
$$X_k = \sum_{m=0}^{N/2-1} x_{2m} e^{-2\pi i k(2m)/N} + \sum_{m=0}^{N/2-1} x_{2m+1} e^{-2\pi i k(2m+1)/N}$$

**Complexity:**
- Naive DFT: $O(N^2)$
- FFT: $O(N \log N)$

**Implementation:** Use library (FFTW, NumPy) for efficiency.

### Grid Setup

**Log-strike grid:**
$$k_j = k_0 + j\Delta k, \quad j = 0, 1, \ldots, N-1$$

**Frequency grid:**
$$u_n = n\Delta u, \quad n = 0, 1, \ldots, N-1$$

**Relationship:**
$$\Delta u \Delta k = \frac{2\pi}{N}$$

**Range:** $k \in [k_0, k_0 + N\Delta k]$

## Carr-Madan Formula

### Option Pricing via Fourier Transform

**European call price:**
$$C(K) = e^{-rT} E[\max(S_T - K, 0)]$$

**Modified call price:**
$$c_T(k) = e^{\alpha k} C_T(k)$$

where $k = \ln(K)$ and $\alpha > 0$ is damping factor.

**Fourier transform:**
$$\psi_T(u) = \int_{-\infty}^{\infty} e^{iuk} c_T(k) dk$$

**Result:**
$$\psi_T(u) = \frac{e^{-rT} \phi_{X_T}(u - (\alpha+1)i)}{(\alpha + iu)(\alpha + 1 + iu)}$$

**Inverse transform:**
$$C_T(k) = \frac{e^{-\alpha k}}{\pi} \int_0^{\infty} \text{Re}[e^{-iuk} \psi_T(u)] du$$

### FFT Implementation

**Discretization:**
$$C_T(k_j) \approx \frac{e^{-\alpha k_j}}{\pi} \sum_{n=0}^{N-1} \text{Re}[e^{-iuk_n} \psi_T(u_n)] \Delta u$$

**Using FFT:**
1. Compute $\psi_T(u_n)$ for $n = 0, \ldots, N-1$
2. Apply FFT to get $C_T(k_j)$
3. Adjust for damping: multiply by $e^{-\alpha k_j}$

**Damping factor:** Typically $\alpha = 1.5$ (ensures integrability).

## COS Method

### Cosine Expansion

**Fang-Oosterlee (2008) COS method:**

Expand density in cosine series:
$$f(x) = \sum_{j=0}^{\infty} A_j \cos\left(j\pi \frac{x-a}{b-a}\right)$$

**Coefficients:**
$$A_j = \frac{2}{b-a} \int_a^b f(x) \cos\left(j\pi \frac{x-a}{b-a}\right) dx$$

**Via characteristic function:**
$$A_j = \frac{2}{b-a} \text{Re}\left[\phi\left(\frac{j\pi}{b-a}\right) \exp\left(-i\frac{aj\pi}{b-a}\right)\right]$$

### Option Pricing

**Call option:**
$$C(K) = e^{-rT} \int_{\ln K}^{\infty} (e^x - K) f(x) dx$$

**Using cosine expansion:**
$$C(K) \approx e^{-rT} \sum_{j=0}^{M-1} A_j V_j$$

where:
$$V_j = \int_{\ln K}^{b} (e^x - K) \cos\left(j\pi \frac{x-a}{b-a}\right) dx$$

**Analytical expression:**
$$V_j = \frac{2}{b-a} \left[\chi_j(\ln K, b) - K \psi_j(\ln K, b)\right]$$

with closed-form $\chi_j$ and $\psi_j$.

### Advantages

- Fast convergence (exponential for smooth densities)
- No damping factor needed
- Accurate for many models

## Lewis Formula

### General Formula

**Option price:**
$$C(K) = S_0 - \frac{Ke^{-rT}}{2\pi} \int_{-\infty}^{\infty} \frac{e^{-iuk} \phi_{X_T}(u-i)}{iu(u-i)} du$$

where $k = \ln(K/S_0)$.

**Simplification:** For $u \in \mathbb{R}$:
$$C(K) = S_0 - \frac{Ke^{-rT}}{2\pi} \int_0^{\infty} \text{Re}\left[\frac{e^{-iuk} \phi_{X_T}(u-i)}{iu(u-i)}\right] du$$

### Implementation

**Discretization:**
$$C(K) \approx S_0 - \frac{Ke^{-rT}}{\pi} \sum_{n=0}^{N-1} \text{Re}\left[\frac{e^{-iu_n k} \phi_{X_T}(u_n-i)}{iu_n(u_n-i)}\right] \Delta u$$

**FFT:** Can use FFT for multiple strikes simultaneously.

## Calibration Using Fourier Methods

### Objective

Find parameters $\boldsymbol{\theta}$ to match market prices:
$$\min_{\boldsymbol{\theta}} \sum_i w_i (P_i^{model}(\boldsymbol{\theta}) - P_i^{market})^2$$

### Gradient Computation

**Via adjoint method:**
$$\frac{\partial P}{\partial \theta_j} = \frac{e^{-rT}}{\pi} \int_0^{\infty} \text{Re}\left[e^{-iuk} \frac{\partial \phi_{X_T}}{\partial \theta_j}(u)\right] du$$

**Efficient:** Compute gradient with one additional FFT per parameter.

### Applications

**Models:**
- Heston stochastic volatility
- Variance Gamma (VG)
- CGMY (tempered stable)
- Jump-diffusion models

**Advantages:**
- Fast pricing (FFT)
- Accurate gradients
- Handles many strikes simultaneously

## Fractional FFT

### Motivation

Standard FFT gives strikes on grid: $k_j = k_0 + j\Delta k$.

**Problem:** May not align with desired strikes.

**Solution:** Fractional FFT allows arbitrary strike spacing.

### Algorithm

**Chirp Z-transform:**
$$X_k = \sum_{n=0}^{N-1} x_n z_n^{-k}$$

where $z_n = e^{2\pi i \beta n}$ with $\beta$ controlling spacing.

**Implementation:** Via convolution (three FFTs).

**Advantages:**
- Fine-grained strike grids
- Align with market strikes
- Similar complexity to FFT

## Example: European Call Pricing

Price European call:
- $S_0 = 100$, $K = 105$, $r = 0.05$, $\sigma = 0.2$, $T = 1$

**Carr-Madan method:**

**Characteristic function:**
$$\phi_{X_T}(u) = \exp\left(iu \times 0.03 - \frac{1}{2}u^2 \times 0.04\right)$$

**Modified transform:**
$$\psi_T(u) = \frac{e^{-0.05} \phi_{X_T}(u - 2.5i)}{(1.5 + iu)(2.5 + iu)}$$

**FFT setup:**
- $N = 2^{12} = 4096$
- $\Delta u = 0.01$
- $\Delta k = 2\pi/(N \Delta u) = 0.153$
- $k_0 = \ln(50) = 3.912$ (covers strikes 50-650)

**Result:**
- Price at $K = 105$: $\$8.02$
- Black-Scholes: $\$8.02$
- Error: $< 0.01\%$

**Computation time:** ~1ms for all 4096 strikes simultaneously.

**Advantage:** Can price entire volatility surface quickly.
