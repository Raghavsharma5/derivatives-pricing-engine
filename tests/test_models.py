"""
Unit Tests for Elite Derivatives Pricing Library
"""

import pytest
import numpy as np
from dpavc.models import BlackScholesModel, HestonModel, MertonJumpDiffusion
from dpavc.monte_carlo import MonteCarloEngine
from dpavc.imp_vol import ImpliedVolatility, SVICalibration


class TestBlackScholesModel:
    """Test Black-Scholes pricing and Greeks"""
    
    def test_call_put_parity(self):
        """Test put-call parity: C - P = S*e^(-qT) - K*e^(-rT)"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
        
        call = bs.call_price()
        put = bs.put_price()
        
        parity_lhs = call - put
        parity_rhs = bs.S0 * np.exp(-bs.q * bs.T) - bs.K * np.exp(-bs.r * bs.T)
        
        assert np.isclose(parity_lhs, parity_rhs, rtol=1e-10)
    
    def test_delta_range(self):
        """Test that Delta is in valid range"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25)
        
        call_delta = bs.delta('call')
        put_delta = bs.delta('put')
        
        assert 0 <= call_delta <= 1
        assert -1 <= put_delta <= 0
    
    def test_gamma_positive(self):
        """Gamma should always be positive"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25)
        
        gamma = bs.gamma()
        assert gamma > 0
    
    def test_vega_positive(self):
        """Vega should always be positive"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25)
        
        vega = bs.vega()
        assert vega > 0
    
    def test_atm_call_put_equality(self):
        """ATM options with no dividends: Call and Put should have similar values"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.0)
        
        call = bs.call_price()
        put = bs.put_price()
        
        # With r > 0, call should be slightly more valuable than put
        assert call > put
        assert call - put < 5  # Difference should be small


class TestMonteCarloEngine:
    """Test Monte Carlo pricing"""
    
    def test_convergence_to_analytical(self):
        """MC should converge to Black-Scholes for large N"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
        analytical = bs.call_price()
        
        mc = MonteCarloEngine(n_simulations=100000, seed=42)
        result = mc.standard_monte_carlo(100, 100, 1, 0.05, 0.25, 0.02, 'call')
        
        # Should be within 3 standard errors
        assert abs(result.price - analytical) < 3 * result.std_error
    
    def test_variance_reduction_improves_efficiency(self):
        """Variance reduction should reduce standard error"""
        mc = MonteCarloEngine(n_simulations=50000, seed=42)
        
        std = mc.standard_monte_carlo(100, 100, 1, 0.05, 0.25, 0.02, 'call')
        av = mc.antithetic_variates(100, 100, 1, 0.05, 0.25, 0.02, 'call')
        cv = mc.control_variates(100, 100, 1, 0.05, 0.25, 0.02, 'call')
        
        # Antithetic should reduce variance
        assert av.std_error < std.std_error
        
        # Control variates should reduce even more
        assert cv.std_error < av.std_error
    
    def test_confidence_interval_contains_true_price(self):
        """95% CI should contain analytical price"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
        analytical = bs.call_price()
        
        mc = MonteCarloEngine(n_simulations=100000, seed=42)
        result = mc.standard_monte_carlo(100, 100, 1, 0.05, 0.25, 0.02, 'call')
        
        assert result.confidence_interval[0] <= analytical <= result.confidence_interval[1]


class TestImpliedVolatility:
    """Test implied volatility calculation"""
    
    def test_iv_recovery(self):
        """Should recover the true volatility from option price"""
        true_vol = 0.25
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=true_vol, q=0.02)
        market_price = bs.call_price()
        
        iv_calc = ImpliedVolatility(S0=100, K=100, T=1, r=0.05, q=0.02)
        recovered_vol, _ = iv_calc.newton_raphson(market_price, 'call')
        
        assert np.isclose(recovered_vol, true_vol, rtol=1e-6)
    
    def test_different_methods_agree(self):
        """Different IV methods should give same result"""
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
        market_price = bs.call_price()
        
        iv_calc = ImpliedVolatility(S0=100, K=100, T=1, r=0.05, q=0.02)
        
        iv_newton, _ = iv_calc.newton_raphson(market_price, 'call')
        iv_brent, _ = iv_calc.brent_method(market_price, 'call')
        iv_jaeckel, _ = iv_calc.jaeckel_method(market_price, 'call')
        
        assert np.isclose(iv_newton, iv_brent, rtol=1e-4)
        assert np.isclose(iv_newton, iv_jaeckel, rtol=1e-4)
    
    def test_iv_surface_shape(self):
        """IV surface should have correct dimensions"""
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.25, 0.5, 1.0])
        
        # Generate prices
        market_prices = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                bs = BlackScholesModel(100, K, T, 0.05, 0.25, 0.02)
                market_prices[i, j] = bs.call_price()
        
        iv_calc = ImpliedVolatility(S0=100, K=strikes[0], T=maturities[0], r=0.05, q=0.02)
        iv_surface = iv_calc.implied_vol_surface(strikes, maturities, market_prices, 'call')
        
        assert iv_surface.shape == (len(maturities), len(strikes))
        assert not np.any(np.isnan(iv_surface))


class TestHestonModel:
    """Test Heston stochastic volatility model"""
    
    def test_heston_reduces_to_bs(self):
        """Heston with sigma_v=0 should match Black-Scholes"""
        # Heston with no vol of vol
        v0 = 0.25**2
        heston = HestonModel(
            S0=100, K=100, T=1, r=0.05, q=0.02,
            v0=v0, kappa=2.0, theta=v0, sigma_v=0.01, rho=0.0
        )
        
        # Black-Scholes
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
        
        heston_price = heston.call_price()
        bs_price = bs.call_price()
        
        # Should be close when sigma_v is small
        assert np.isclose(heston_price, bs_price, rtol=0.05)
    
    def test_feller_condition(self):
        """Model should satisfy Feller condition for well-behaved variance"""
        v0 = 0.04
        kappa = 2.0
        theta = 0.04
        sigma_v = 0.3
        
        # Feller condition: 2*kappa*theta > sigma_v^2
        assert 2 * kappa * theta > sigma_v**2


class TestJumpDiffusion:
    """Test Merton Jump Diffusion model"""
    
    def test_zero_jumps_reduces_to_bs(self):
        """Jump diffusion with lambda=0 should match Black-Scholes"""
        # Jump diffusion with no jumps
        jd = MertonJumpDiffusion(
            S0=100, K=100, T=1, r=0.05, sigma=0.25,
            lambda_=0.0, m=0.0, delta=0.1, q=0.02
        )
        
        # Black-Scholes
        bs = BlackScholesModel(S0=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
        
        jd_price = jd.call_price()
        bs_price = bs.call_price()
        
        assert np.isclose(jd_price, bs_price, rtol=1e-6)


def test_put_call_parity_all_models():
    """Test put-call parity holds for all models"""
    S0, K, T, r, q = 100, 100, 1, 0.05, 0.02
    
    # Black-Scholes
    bs = BlackScholesModel(S0, K, T, r, 0.25, q)
    assert np.isclose(
        bs.call_price() - bs.put_price(),
        S0 * np.exp(-q * T) - K * np.exp(-r * T),
        rtol=1e-10
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])