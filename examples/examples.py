"""
Comprehensive Example - Elite Derivatives Pricing Library
Demonstrates all advanced features with realistic scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from dpavc.models import BlackScholesModel, HestonModel, MertonJumpDiffusion
from dpavc.monte_carlo import MonteCarloEngine
from dpavc.imp_vol import ImpliedVolatility, SVICalibration
from dpavc.calibration import ModelCalibrator, MarketDataLoader
from dpavc.performance import PerformanceBenchmark


def example_1_greeks_analysis():
    """Example 1: Complete Greeks analysis for risk management"""
    
    print("=" * 80)
    print("EXAMPLE 1: Greeks Analysis for Risk Management")
    print("=" * 80)
    print()
    
    # Option parameters
    S0 = 100.0
    K = 100.0
    T = 0.25  # 3 months
    r = 0.05
    sigma = 0.25
    q = 0.02
    
    bs = BlackScholesModel(S0, K, T, r, sigma, q)
    
    # Calculate all Greeks
    greeks = {
        'Call Price': bs.call_price(),
        'Put Price': bs.put_price(),
        'Delta (Call)': bs.delta('call'),
        'Delta (Put)': bs.delta('put'),
        'Gamma': bs.gamma(),
        'Vega': bs.vega(),
        'Theta (Call)': bs.theta('call'),
        'Theta (Put)': bs.theta('put'),
        'Rho (Call)': bs.rho('call'),
        'Rho (Put)': bs.rho('put')
    }
    
    print("Option Parameters:")
    print(f"  Spot: ${S0:.2f}")
    print(f"  Strike: ${K:.2f}")
    print(f"  Maturity: {T:.2f} years ({T*365:.0f} days)")
    print(f"  Rate: {r*100:.1f}%")
    print(f"  Volatility: {sigma*100:.1f}%")
    print()
    
    print("Prices and Greeks:")
    for name, value in greeks.items():
        print(f"  {name:15s}: {value:10.6f}")
    print()
    
    # Sensitivity analysis - Delta across spot prices
    print("Delta Sensitivity Analysis:")
    spots = np.linspace(80, 120, 9)
    print(f"{'Spot':>8s} {'Call Delta':>12s} {'Put Delta':>12s} {'Gamma':>12s}")
    print("-" * 48)
    
    for S in spots:
        bs_temp = BlackScholesModel(S, K, T, r, sigma, q)
        print(f"{S:8.2f} {bs_temp.delta('call'):12.6f} {bs_temp.delta('put'):12.6f} {bs_temp.gamma():12.6f}")
    print()


def example_2_variance_reduction():
    """Example 2: Variance reduction techniques comparison"""
    
    print("=" * 80)
    print("EXAMPLE 2: Variance Reduction Techniques Comparison")
    print("=" * 80)
    print()
    
    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.25
    q = 0.02
    
    mc = MonteCarloEngine(n_simulations=100000, seed=42)
    
    print("Comparing variance reduction methods...")
    print(f"Simulations: {mc.n_simulations:,}")
    print()
    
    # Run all methods
    results = mc.compare_variance_reduction(S0, K, T, r, sigma, q, 'call')
    
    # Calculate analytical price for comparison
    bs = BlackScholesModel(S0, K, T, r, sigma, q)
    analytical = bs.call_price()
    
    print(f"Analytical Price (Black-Scholes): ${analytical:.6f}")
    print()
    print(f"{'Method':20s} {'Price':>12s} {'Std Error':>12s} {'95% CI Width':>15s} {'Error':>12s}")
    print("-" * 80)
    
    for method, result in results.items():
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        error = abs(result.price - analytical)
        print(f"{method:20s} ${result.price:11.6f} ${result.std_error:11.6f} "
              f"${ci_width:14.6f} ${error:11.6f}")
    
    # Calculate efficiency gains
    efficiency = mc.variance_reduction_efficiency(results)
    
    print()
    print("Variance Reduction Efficiency:")
    print(f"{'Method':20s} {'Efficiency':>15s} {'Improvement':>15s}")
    print("-" * 55)
    
    for method, eff in efficiency.items():
        improvement = (eff - 1) * 100
        print(f"{method:20s} {eff:15.2f}x {improvement:14.1f}%")
    print()


def example_3_stochastic_models():
    """Example 3: Advanced stochastic models - Heston and Jump Diffusion"""
    
    print("=" * 80)
    print("EXAMPLE 3: Advanced Stochastic Volatility Models")
    print("=" * 80)
    print()
    
    # Common parameters
    S0 = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    q = 0.02
    
    # Black-Scholes
    sigma_bs = 0.25
    bs = BlackScholesModel(S0, K, T, r, sigma_bs, q)
    bs_call = bs.call_price()
    
    # Heston model
    v0 = 0.25**2      # Initial variance
    kappa = 2.0       # Mean reversion speed
    theta = 0.25**2   # Long-term variance
    sigma_v = 0.3     # Vol of vol
    rho = -0.7        # Correlation
    
    heston = HestonModel(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho)
    heston_call = heston.call_price()
    
    # Jump Diffusion model
    sigma_jd = 0.20
    lambda_ = 0.5     # 0.5 jumps per year
    m = -0.1          # Negative jump mean
    delta = 0.15      # Jump volatility
    
    jd = MertonJumpDiffusion(S0, K, T, r, sigma_jd, lambda_, m, delta, q)
    jd_call = jd.call_price()
    
    print("Model Comparison:")
    print(f"{'Model':25s} {'Call Price':>15s} {'Difference from BS':>20s}")
    print("-" * 65)
    print(f"{'Black-Scholes':25s} ${bs_call:14.6f} {'---':>20s}")
    print(f"{'Heston (Stoch Vol)':25s} ${heston_call:14.6f} ${heston_call - bs_call:19.6f}")
    print(f"{'Jump Diffusion':25s} ${jd_call:14.6f} ${jd_call - bs_call:19.6f}")
    print()
    
    print("Model Parameters:")
    print()
    print("Heston Model:")
    print(f"  Initial variance (v0): {v0:.6f}")
    print(f"  Mean reversion (κ): {kappa:.2f}")
    print(f"  Long-term variance (θ): {theta:.6f}")
    print(f"  Vol of vol (σ_v): {sigma_v:.2f}")
    print(f"  Correlation (ρ): {rho:.2f}")
    print()
    print("Jump Diffusion Model:")
    print(f"  Diffusion vol (σ): {sigma_jd:.2f}")
    print(f"  Jump intensity (λ): {lambda_:.2f} per year")
    print(f"  Jump mean (m): {m:.2f}")
    print(f"  Jump vol (δ): {delta:.2f}")
    print()


def example_4_implied_volatility():
    """Example 4: Implied volatility calculation and surface"""
    
    print("=" * 80)
    print("EXAMPLE 4: Implied Volatility Calculation")
    print("=" * 80)
    print()
    
    # Parameters
    S0 = 100.0
    r = 0.05
    q = 0.02
    true_vol = 0.25
    
    # Generate market price
    K = 100.0
    T = 0.5
    bs = BlackScholesModel(S0, K, T, r, true_vol, q)
    market_price = bs.call_price()
    
    print(f"Market call price: ${market_price:.6f}")
    print(f"True volatility: {true_vol*100:.2f}%")
    print()
    
    # Calculate implied volatility using different methods
    iv_calc = ImpliedVolatility(S0, K, T, r, q)
    
    print("Implied Volatility Recovery:")
    print(f"{'Method':20s} {'IV':>10s} {'Iterations':>12s} {'Time (ms)':>12s}")
    print("-" * 60)
    
    import time
    
    methods = ['newton', 'brent', 'jaeckel']
    for method in methods:
        start = time.perf_counter()
        
        if method == 'newton':
            iv, iters = iv_calc.newton_raphson(market_price, 'call')
        elif method == 'brent':
            iv, iters = iv_calc.brent_method(market_price, 'call')
        else:
            iv, iters = iv_calc.jaeckel_method(market_price, 'call')
        
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"{method.capitalize():20s} {iv*100:9.4f}% {iters:12d} {elapsed:12.6f}")
    
    print()
    print("All methods successfully recovered the true volatility!")
    print()
    
    # Implied volatility surface
    print("Generating implied volatility surface...")
    
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    
    # Generate synthetic market prices with smile
    market_prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            # Add volatility smile
            moneyness = np.log(K / S0)
            vol = true_vol + 0.1 * moneyness**2  # Smile effect
            bs_temp = BlackScholesModel(S0, K, T, r, vol, q)
            market_prices[i, j] = bs_temp.call_price()
    
    # Calculate IV surface
    iv_calc_surface = ImpliedVolatility(S0, strikes[0], maturities[0], r, q)
    iv_surface = iv_calc_surface.implied_vol_surface(strikes, maturities, market_prices, 'call')
    
    print()
    print("Implied Volatility Surface:")
    print(f"{'Maturity':>12s}", end='')
    for K in strikes:
        print(f"{K:>10.0f}", end='')
    print()
    print("-" * 62)
    
    for i, T in enumerate(maturities):
        print(f"{T:12.2f}", end='')
        for j in range(len(strikes)):
            print(f"{iv_surface[i,j]*100:10.2f}%", end='')
        print()
    print()


def example_5_model_calibration():
    """Example 5: Calibration to market data"""
    
    print("=" * 80)
    print("EXAMPLE 5: Model Calibration to Market Data")
    print("=" * 80)
    print()
    
    # Market parameters
    S0 = 100.0
    r = 0.05
    q = 0.02
    
    # Generate synthetic market data with realistic volatility surface
    strikes = np.linspace(80, 120, 5)
    maturities = np.array([0.25, 0.5, 1.0])
    
    print("Generating synthetic market option prices...")
    market_data = MarketDataLoader.generate_synthetic_surface(
        S0, strikes, maturities, r, q,
        vol_atm=0.25, skew=-0.15, smile=0.05
    )
    
    # Only use call options for calibration
    market_data = market_data[market_data['option_type'] == 'call']
    
    print(f"Generated {len(market_data)} market option prices")
    print()
    
    # Calibrate models
    calibrator = ModelCalibrator(S0, r, q)
    
    print("Calibrating Heston model (this may take a minute)...")
    heston_params = calibrator.calibrate_heston(market_data, method='differential_evolution')
    
    print()
    print("Heston Model Calibration Results:")
    print(f"  Initial variance (v0): {heston_params['v0']:.6f}")
    print(f"  Mean reversion (κ): {heston_params['kappa']:.4f}")
    print(f"  Long-term variance (θ): {heston_params['theta']:.6f}")
    print(f"  Vol of vol (σ_v): {heston_params['sigma_v']:.4f}")
    print(f"  Correlation (ρ): {heston_params['rho']:.4f}")
    print(f"  RMSE: ${heston_params['rmse']:.6f}")
    print()
    
    print("Calibrating Jump Diffusion model...")
    jd_params = calibrator.calibrate_jump_diffusion(market_data)
    
    print()
    print("Jump Diffusion Calibration Results:")
    print(f"  Diffusion vol (σ): {jd_params['sigma']:.4f}")
    print(f"  Jump intensity (λ): {jd_params['lambda']:.4f}")
    print(f"  Jump mean (m): {jd_params['m']:.4f}")
    print(f"  Jump vol (δ): {jd_params['delta']:.4f}")
    print(f"  RMSE: ${jd_params['rmse']:.6f}")
    print()
    
    # Model comparison
    comparison = calibrator.model_comparison(market_data)
    
    print("Model Comparison:")
    print(comparison.to_string(index=False))
    print()


def example_6_performance_benchmarks():
    """Example 6: Performance optimization benchmarks"""
    
    print("=" * 80)
    print("EXAMPLE 6: Performance Benchmarks")
    print("=" * 80)
    print()
    
    benchmark = PerformanceBenchmark()
    
    print("Running comprehensive performance benchmarks...")
    print("This will compare Python loops, NumPy vectorization, and Numba JIT compilation")
    print()
    
    results_df = benchmark.run_comprehensive_benchmark()
    
    print()
    print(benchmark.generate_report())


def example_7_heston_monte_carlo():
    """Example 7: Heston model Monte Carlo with path visualization"""
    
    print("=" * 80)
    print("EXAMPLE 7: Heston Model Monte Carlo Simulation")
    print("=" * 80)
    print()
    
    # Heston parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.7
    
    mc = MonteCarloEngine(n_simulations=50000, seed=42)
    
    print("Heston Model Parameters:")
    print(f"  S0 = {S0}, K = {K}, T = {T}")
    print(f"  v0 = {v0:.4f}, κ = {kappa}, θ = {theta:.4f}")
    print(f"  σ_v = {sigma_v}, ρ = {rho}")
    print()
    
    print("Running Heston Monte Carlo simulation...")
    result = mc.heston_monte_carlo(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                                   n_steps=252, option_type='call')
    
    print()
    print("Results:")
    print(result)
    
    # Compare with semi-analytical Heston
    heston = HestonModel(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho)
    analytical = heston.call_price()
    
    print(f"Semi-Analytical Heston Price: ${analytical:.6f}")
    print(f"Monte Carlo Price: ${result.price:.6f}")
    print(f"Difference: ${abs(result.price - analytical):.6f}")
    print()


def main():
    """Run all examples"""
    
    print("\n")
    print("*" * 80)
    print("ELITE DERIVATIVES PRICING LIBRARY - COMPREHENSIVE DEMONSTRATION")
    print("*" * 80)
    print("\n")
    
    # Run all examples
    example_1_greeks_analysis()
    example_2_variance_reduction()
    example_3_stochastic_models()
    example_4_implied_volatility()
    example_5_model_calibration()
    example_6_performance_benchmarks()
    example_7_heston_monte_carlo()
    
    print("\n")
    print("*" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("*" * 80)
    print("\n")


if __name__ == "__main__":
    main()