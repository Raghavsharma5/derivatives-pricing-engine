#!/usr/bin/env python3
"""
Quick Demo - Elite Derivatives Pricing Library
"""

import numpy as np
import pandas as pd
from dpavc.models import BlackScholesModel, HestonModel, MertonJumpDiffusion
from dpavc.monte_carlo import MonteCarloEngine
from dpavc.imp_vol import ImpliedVolatility
from dpavc.calibration import MarketDataLoader

print("=" * 80)
print("ELITE DERIVATIVES PRICING LIBRARY - QUICK DEMO")
print("=" * 80)
print()

# 1. BLACK-SCHOLES WITH GREEKS
print("1. Black-Scholes Model with Complete Greeks")
print("-" * 80)

bs = BlackScholesModel(S0=100, K=100, T=0.25, r=0.05, sigma=0.25, q=0.02)

greeks = {
    'Call Price': bs.call_price(),
    'Put Price': bs.put_price(),
    'Delta (Call)': bs.delta('call'),
    'Gamma': bs.gamma(),
    'Vega': bs.vega(),
    'Theta (Call)': bs.theta('call'),
    'Rho (Call)': bs.rho('call')
}

for name, value in greeks.items():
    print(f"  {name:15s}: {value:10.6f}")
print()

# 2. VARIANCE REDUCTION
print("2. Monte Carlo Variance Reduction Techniques")
print("-" * 80)

mc = MonteCarloEngine(n_simulations=50000, seed=42)

print("Comparing variance reduction methods (50,000 paths)...")
results = mc.compare_variance_reduction(100, 100, 1, 0.05, 0.25, 0.02, 'call')

analytical = bs_annual = BlackScholesModel(100, 100, 1, 0.05, 0.25, 0.02).call_price()
print(f"\nAnalytical Price: ${analytical:.6f}")
print(f"\n{'Method':20s} {'Price':>12s} {'Std Error':>12s} {'Error':>12s}")
print("-" * 60)

for method, result in results.items():
    error = abs(result.price - analytical)
    print(f"{method:20s} ${result.price:11.6f} ${result.std_error:11.6f} ${error:11.6f}")

efficiency = mc.variance_reduction_efficiency(results)
print("\nVariance Reduction Efficiency:")
for method, eff in efficiency.items():
    print(f"  {method}: {eff:.2f}x reduction")
print()

# 3. IMPLIED VOLATILITY
print("3. Implied Volatility Calculation")
print("-" * 80)

true_vol = 0.25
bs_iv = BlackScholesModel(100, 100, 0.5, 0.05, true_vol, 0.02)
market_price = bs_iv.call_price()

iv_calc = ImpliedVolatility(100, 100, 0.5, 0.05, 0.02)

print(f"Market Price: ${market_price:.6f}")
print(f"True Volatility: {true_vol*100:.2f}%")
print()

methods = [('Newton-Raphson', 'newton'), ('Brent', 'brent'), ('Jaeckel', 'jaeckel')]
print(f"{'Method':20s} {'IV':>10s} {'Iterations':>12s}")
print("-" * 45)

for name, method in methods:
    if method == 'newton':
        iv, iters = iv_calc.newton_raphson(market_price, 'call')
    elif method == 'brent':
        iv, iters = iv_calc.brent_method(market_price, 'call')
    else:
        iv, iters = iv_calc.jaeckel_method(market_price, 'call')
    print(f"{name:20s} {iv*100:9.4f}% {iters:12d}")
print()

# 4. ADVANCED MODELS
print("4. Advanced Stochastic Models")
print("-" * 80)

# Heston
heston = HestonModel(
    S0=100, K=100, T=0.5, r=0.05, q=0.02,
    v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7
)
heston_price = heston.call_price()

# Jump Diffusion
jd = MertonJumpDiffusion(
    S0=100, K=100, T=0.5, r=0.05, sigma=0.20,
    lambda_=0.5, m=-0.1, delta=0.15, q=0.02
)
jd_price = jd.call_price()

bs_comp = BlackScholesModel(100, 100, 0.5, 0.05, 0.25, 0.02)
bs_price = bs_comp.call_price()

print(f"{'Model':25s} {'Call Price':>15s}")
print("-" * 45)
print(f"{'Black-Scholes':25s} ${bs_price:14.6f}")
print(f"{'Heston (Stoch Vol)':25s} ${heston_price:14.6f}")
print(f"{'Jump Diffusion':25s} ${jd_price:14.6f}")
print()

# 5. IMPLIED VOLATILITY SURFACE
print("5. Implied Volatility Surface")
print("-" * 80)

strikes = np.array([90, 95, 100, 105, 110])
maturities = np.array([0.25, 0.5, 1.0])

print("Generating synthetic market data...")
market_data = MarketDataLoader.generate_synthetic_surface(
    100, strikes, maturities, r=0.05, q=0.02,
    vol_atm=0.25, skew=-0.15, smile=0.05
)

# Extract implied vols
print("\nImplied Volatility Surface:")
print(f"{'Maturity':>12s}", end='')
for K in strikes:
    print(f"{K:>10.0f}", end='')
print()
print("-" * 62)

for T in maturities:
    print(f"{T:12.2f}", end='')
    subset = market_data[(market_data['maturity'] == T) & (market_data['option_type'] == 'call')]
    for K in strikes:
        iv = subset[subset['strike'] == K]['implied_vol'].values[0]
        print(f"{iv*100:10.2f}%", end='')
    print()
print()

print("=" * 80)
print("DEMO COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nFor comprehensive examples, run: python examples.py")
print("For unit tests, run: pytest test_models.py")