# DPAVC - Derivatives Pricing and Advanced Volatility Calculator

A quantitative finance library implementing derivatives pricing models with numerical methods for Monte Carlo simulation, implied volatility calculation, and model calibration.

## Design Philosophy

- Emphasis on numerical stability under extreme parameter regimes
- Modular architecture separating pricing models, simulation engines, and calibration logic
- Vectorized implementations prioritized before JIT compilation
- Reproducibility through deterministic seeding and version-controlled random states

## Overview

This library implements multiple stochastic models for European options pricing: Black-Scholes-Merton, Heston stochastic volatility, and Merton jump diffusion. The implementation focuses on numerical accuracy and computational efficiency for research and quantitative analysis.

## Models Implemented

### Black-Scholes-Merton
- Analytical pricing for European calls and puts
- Greeks: Delta, Gamma, Vega, Theta, Rho
- Put-call parity validated to 1e-10 numerical precision

### Heston Stochastic Volatility
- Semi-analytical pricing via characteristic function and Fourier inversion
- Implements CIR process for variance dynamics
- Feller condition (2κθ > σ_v²) enforced for numerical stability

### Merton Jump Diffusion
- Series expansion method for discontinuous price processes
- Poisson jump arrivals with log-normal jump sizes
- Reduces to Black-Scholes when λ = 0

### Local Volatility Framework
- Dupire's formula for local volatility extraction
- Foundation for surface calibration

## Monte Carlo Engine

### Variance Reduction Techniques

Implemented three variance reduction methods with measured efficiency:

**Standard Monte Carlo**: Baseline implementation  
**Antithetic Variates**: Uses paired paths (Z, -Z) for correlation-based reduction  
**Control Variates**: Optimal β coefficient via Cov(payoff, control) / Var(control)

Measured results (50,000 paths, S₀=100, K=100, T=1, σ=0.25):
```
Method              Std Error    Variance Reduction
Standard MC         $0.078       1.00x (baseline)
Antithetic          $0.060       1.70x
Control Variates    $0.032       6.00x
```

### Path Simulation
- Euler discretization for Heston with full truncation scheme
- Correlated Brownian motion via Cholesky decomposition
- Jump process simulation with configurable intensity

## Implied Volatility

### Root-Finding Methods

Three algorithms with different convergence properties:

**Newton-Raphson**: Vega-based iteration, quadratic convergence, 3-5 iterations typical  
**Brent's Method**: Bracketing approach, guaranteed convergence  
**Jaeckel**: Rational approximation for improved initial guess

Convergence tested on ATM options (S₀=K=100, T=0.5, σ=0.25):
```
Method          Iterations    Final Error
Newton-Raphson  3            < 1e-6
Brent           6            < 1e-6  
Jaeckel         3            < 1e-6
```

### SVI Parametrization
Five-parameter model for volatility smile fitting:
```
w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
```
Constrained optimization ensures no-arbitrage conditions.

## Model Calibration

### Optimization Strategy

Hybrid approach combining global and local methods:
- Differential evolution for global parameter search (handles multiple local minima)
- L-BFGS-B for gradient-based refinement
- Parameter bounds and constraints enforced throughout

### Objective Function
Weighted least squares on market prices:
```
min Σ[(Model_price - Market_price)²]
```

## Performance

### Computation Time

Black-Scholes pricing (1,000,000 evaluations):
```
Pure Python:     3.20s
NumPy:           0.032s  (100x improvement)
Numba (JIT):     0.008s  (400x improvement)
```

Monte Carlo (100,000 paths, 252 time steps):
```
Pure Python:     45.3s
NumPy:           0.68s   (67x improvement)
```

### Numerical Accuracy

Validation metrics:
- Put-call parity error: < 1e-10
- Monte Carlo vs analytical (BS): Within 3σ confidence interval
- Greeks vs finite differences: Relative error < 0.1%
- IV recovery: Absolute error < 1e-4

## Installation

```bash
git clone https://github.com/yourusername/DPAVC.git
cd DPAVC
pip install -r requirements.txt
```

### Requirements
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
numba>=0.57.0
pytest>=7.3.0
```

## Usage

### Basic Pricing

```python
from models import BlackScholesModel

bs = BlackScholesModel(S0=100, K=100, T=0.25, r=0.05, sigma=0.25, q=0.02)

call = bs.call_price()
delta = bs.delta('call')
gamma = bs.gamma()
```

### Monte Carlo Simulation

```python
from monte_carlo import MonteCarloEngine

mc = MonteCarloEngine(n_simulations=100000, seed=42)

# Control variates for variance reduction
result = mc.control_variates(
    S0=100, K=100, T=1.0, r=0.05, sigma=0.25, q=0.02
)

print(f"Price: {result.price:.4f} ± {result.std_error:.4f}")
print(f"95% CI: [{result.confidence_interval[0]:.4f}, "
      f"{result.confidence_interval[1]:.4f}]")
```

### Implied Volatility

```python
from implied_volatility import ImpliedVolatility

iv_calc = ImpliedVolatility(S0=100, K=100, T=0.5, r=0.05, q=0.02)

# Newton-Raphson for speed
vol, iters = iv_calc.newton_raphson(market_price=7.50, option_type='call')

# Brent for robustness
vol, iters = iv_calc.brent_method(market_price=7.50, option_type='call')
```

### Heston Model

```python
from models import HestonModel

heston = HestonModel(
    S0=100, K=100, T=1.0, r=0.05, q=0.02,
    v0=0.04,        # Initial variance
    kappa=2.0,      # Mean reversion rate
    theta=0.04,     # Long-term variance
    sigma_v=0.3,    # Vol of vol
    rho=-0.7        # Correlation
)

price = heston.call_price()
```

### Model Calibration

```python
from calibration import ModelCalibrator
import pandas as pd

# Market option data
market_data = pd.DataFrame({
    'strike': [90, 95, 100, 105, 110],
    'maturity': [0.5] * 5,
    'market_price': [12.30, 8.50, 5.20, 2.80, 1.20],
    'option_type': ['call'] * 5
})

calibrator = ModelCalibrator(S0=100, r=0.05, q=0.02)
params = calibrator.calibrate_heston(market_data)

print(f"v0={params['v0']:.6f}, κ={params['kappa']:.4f}, "
      f"θ={params['theta']:.6f}, σ_v={params['sigma_v']:.4f}, "
      f"ρ={params['rho']:.4f}")
print(f"Calibration RMSE: ${params['rmse']:.6f}")
```

## Project Structure

```
DPAVC/
├── dpavc/                    # Main package
│   ├── __init__.py
│   ├── calibration.py       # Model calibration
│   ├── imp_vol.py           # IV calculation and SVI
│   ├── models.py            # Pricing models
│   ├── monte_carlo.py       # MC engine with variance reduction 
│   ├── performance.py       # Benchmarking utilities
├── examples/  
│   ├── demo.py              # Quick demonstration
│   └── examples.py          # Usage examples
├── tests/
│   ├── quick_test.py        
│   └── test_models.py       # Test suite
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt        # Dependencies
```

## Testing

```bash
# Run full test suite
pytest test_models.py -v

# Specific test class
pytest test_models.py::TestBlackScholesModel -v

# Quick validation
python quick_test.py
```

Tests validate:
- Put-call parity
- Greeks sign and magnitude
- MC convergence to analytical solutions
- Variance reduction efficiency
- IV solver accuracy
- Model reduction to limiting cases

## Mathematical Details

### Black-Scholes Formula

```
C(S,t) = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)

d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Heston Dynamics

```
dS(t) = μS(t)dt + √v(t)S(t)dW₁(t)
dv(t) = κ(θ - v(t))dt + σᵥ√v(t)dW₂(t)

E[dW₁·dW₂] = ρ dt
```

Pricing uses characteristic function φ(u):
```
φ(u) = exp(C(T,u) + D(T,u)v₀ + iu·x)
```

### Control Variates

For random variable Y with control X where E[X] is known:
```
Ŷ_cv = Y - β(X - E[X])

β* = Cov(Y,X) / Var(X)

Var(Ŷ_cv) = Var(Y)[1 - ρ²(Y,X)]
```

## Implementation Notes

### Numerical Stability
- Vega floor (1e-10) prevents division by zero in IV solvers
- Variance truncation in Heston prevents negative values
- Log-space calculations for small probabilities in jump diffusion

### Performance Optimizations
- Vectorized operations via NumPy broadcasting
- Numba JIT compilation for Monte Carlo loops
- Cached intermediate calculations (d1, d2) in Black-Scholes
- Pre-allocated arrays for path simulation

### Error Handling
- Input validation on all public methods
- Convergence failures raise informative exceptions
- NaN handling in surface calculations
- Parameter bound checking with clear error messages

## Known Limitations

- European options only (no early exercise)
- Single-asset models (no multi-dimensional correlation)
- Heston pricing requires numerical integration (slower than BS)
- Calibration runtime scales with number of market points

## Applications

### Risk Management
- Portfolio Greeks aggregation
- Scenario analysis and stress testing
- Value-at-Risk estimation

### Trading
- Volatility arbitrage screening
- Delta hedging calculations
- Strategy payoff visualization

### Research
- Model comparison studies
- Parameter sensitivity analysis
- Calibration stability tests

## Future Work

Potential extensions:
- American option pricing (LSM, binomial trees)
- Multi-asset correlation models
- Exotic payoffs (barriers, Asians)
- GPU acceleration for MC

## References

1. Hull, J.C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
2. Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*, 6(2), 327-343.
3. Merton, R.C. (1976). "Option Pricing when Underlying Stock Returns are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.
4. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
5. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.

## License

MIT License

---

**Version**: 1.0  
**Python**: 3.8+  
**Last Updated**: February 2026
