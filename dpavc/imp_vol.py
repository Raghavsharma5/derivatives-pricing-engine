"""
Implied Volatility Calculation with Multiple Root-Finding Methods
Implements Newton-Raphson, Brent's method, and SVI calibration
"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Tuple, Optional, List
import warnings

from dpavc.models import BlackScholesModel


class ImpliedVolatility:
    """Calculate implied volatility using various numerical methods"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, q: float = 0.0):
        """
        Parameters:
        -----------
        S0: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        
    def newton_raphson(self, market_price: float, option_type: str = 'call',
                      initial_guess: float = 0.3, tol: float = 1e-6,
                      max_iter: int = 100) -> Tuple[float, int]:
        """
        Newton-Raphson method using vega as derivative
        Fastest convergence when close to solution
        """
        
        sigma = initial_guess
        
        for i in range(max_iter):
            bs = BlackScholesModel(self.S0, self.K, self.T, self.r, sigma, self.q)
            
            if option_type.lower() == 'call':
                price = bs.call_price()
            else:
                price = bs.put_price()
            
            vega = bs.vega() * 100  # Convert back from per 1% to actual vega
            
            diff = market_price - price
            
            if abs(diff) < tol:
                return sigma, i + 1
            
            if vega < 1e-10:  # Avoid division by zero
                raise ValueError("Vega too small, cannot converge")
            
            sigma = sigma + diff / vega
            
            # Keep sigma positive and reasonable
            sigma = max(0.001, min(sigma, 5.0))
        
        warnings.warn(f"Newton-Raphson did not converge after {max_iter} iterations")
        return sigma, max_iter
    
    def brent_method(self, market_price: float, option_type: str = 'call',
                    sigma_min: float = 0.001, sigma_max: float = 5.0,
                    tol: float = 1e-6) -> Tuple[float, int]:
        """
        Brent's method - very robust but potentially slower
        Guaranteed to converge if solution exists in bracket
        """
        
        def objective(sigma):
            bs = BlackScholesModel(self.S0, self.K, self.T, self.r, sigma, self.q)
            if option_type.lower() == 'call':
                return bs.call_price() - market_price
            else:
                return bs.put_price() - market_price
        
        # Check if solution exists in bracket
        f_min = objective(sigma_min)
        f_max = objective(sigma_max)
        
        if f_min * f_max > 0:
            raise ValueError(f"No root in bracket [{sigma_min}, {sigma_max}]")
        
        # Brent's method
        result = brentq(objective, sigma_min, sigma_max, xtol=tol, full_output=True)
        sigma = result[0]
        iterations = result[1].iterations
        
        return sigma, iterations
    
    def jaeckel_method(self, market_price: float, option_type: str = 'call',
                      tol: float = 1e-6, max_iter: int = 100) -> Tuple[float, int]:
        """
        Jaeckel's method with rational guess
        Better initial guess than standard methods
        """
        
        # Intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(self.S0 * np.exp(-self.q * self.T) - 
                          self.K * np.exp(-self.r * self.T), 0)
        else:
            intrinsic = max(self.K * np.exp(-self.r * self.T) - 
                          self.S0 * np.exp(-self.q * self.T), 0)
        
        # Time value
        time_value = market_price - intrinsic
        
        if time_value <= 0:
            return 0.0, 0  # At or below intrinsic value
        
        # Jaeckel's rational initial guess
        moneyness = np.log(self.S0 / self.K) + (self.r - self.q) * self.T
        normalized_price = time_value / (self.S0 * np.exp(-self.q * self.T))
        
        # Rational approximation
        initial_guess = np.sqrt(2 * np.pi / self.T) * normalized_price
        
        # Refine with Newton-Raphson
        return self.newton_raphson(market_price, option_type, initial_guess, tol, max_iter)
    
    def implied_vol_surface(self, strikes: np.ndarray, maturities: np.ndarray,
                           market_prices: np.ndarray, option_type: str = 'call',
                           method: str = 'newton') -> np.ndarray:
        """
        Calculate implied volatility surface
        
        Parameters:
        -----------
        strikes: Array of strike prices
        maturities: Array of time to maturity
        market_prices: 2D array of market prices [len(maturities) x len(strikes)]
        option_type: 'call' or 'put'
        method: 'newton', 'brent', or 'jaeckel'
        
        Returns:
        --------
        2D array of implied volatilities
        """
        
        iv_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                try:
                    iv_calc = ImpliedVolatility(self.S0, K, T, self.r, self.q)
                    
                    if method == 'newton':
                        iv, _ = iv_calc.newton_raphson(market_prices[i, j], option_type)
                    elif method == 'brent':
                        iv, _ = iv_calc.brent_method(market_prices[i, j], option_type)
                    elif method == 'jaeckel':
                        iv, _ = iv_calc.jaeckel_method(market_prices[i, j], option_type)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    iv_surface[i, j] = iv
                    
                except Exception as e:
                    warnings.warn(f"Failed to compute IV for K={K}, T={T}: {str(e)}")
                    iv_surface[i, j] = np.nan
        
        return iv_surface


class SVICalibration:
    """
    SVI (Stochastic Volatility Inspired) parametrization
    Used for fitting implied volatility smile
    
    Total variance w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = log(K/F) is log-moneyness
    """
    
    def __init__(self):
        self.params = None
        
    @staticmethod
    def svi_variance(k: np.ndarray, a: float, b: float, rho: float, 
                    m: float, sigma: float) -> np.ndarray:
        """SVI total variance formula"""
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def fit(self, log_moneyness: np.ndarray, total_variance: np.ndarray,
           initial_guess: Optional[List[float]] = None) -> dict:
        """
        Calibrate SVI parameters to market implied volatility
        
        Parameters:
        -----------
        log_moneyness: Log-moneyness values k = log(K/F)
        total_variance: Market total variance (sigma^2 * T)
        initial_guess: Initial parameter guess [a, b, rho, m, sigma]
        
        Returns:
        --------
        Dictionary with calibrated parameters
        """
        
        from scipy.optimize import minimize
        
        if initial_guess is None:
            # Reasonable default guess
            initial_guess = [
                np.mean(total_variance),  # a
                0.1,                       # b
                0.0,                       # rho
                0.0,                       # m
                0.1                        # sigma
            ]
        
        def objective(params):
            a, b, rho, m, sigma = params
            
            # Ensure parameter constraints
            if b < 0 or sigma < 0 or abs(rho) >= 1:
                return 1e10
            
            predicted = self.svi_variance(log_moneyness, a, b, rho, m, sigma)
            return np.sum((predicted - total_variance)**2)
        
        # Parameter constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[1]},  # b >= 0
            {'type': 'ineq', 'fun': lambda x: x[4]},  # sigma >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - abs(x[2])}  # |rho| < 1
        ]
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                        constraints=constraints)
        
        if result.success:
            self.params = {
                'a': result.x[0],
                'b': result.x[1],
                'rho': result.x[2],
                'm': result.x[3],
                'sigma': result.x[4]
            }
            return self.params
        else:
            raise ValueError("SVI calibration failed")
    
    def predict(self, log_moneyness: np.ndarray) -> np.ndarray:
        """Predict total variance for given log-moneyness using fitted parameters"""
        if self.params is None:
            raise ValueError("Model must be fitted first")
        
        return self.svi_variance(log_moneyness, **self.params)
    
    def implied_volatility(self, log_moneyness: np.ndarray, T: float) -> np.ndarray:
        """Convert total variance to implied volatility"""
        total_var = self.predict(log_moneyness)
        return np.sqrt(total_var / T)