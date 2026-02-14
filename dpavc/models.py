"""
Advanced Derivatives Pricing Models
Implements Black-Scholes, Heston, Merton Jump Diffusion, and Local Volatility models
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import factorial
from typing import Tuple, Optional
import warnings

class BlackScholesModel:
    """Standard Black-Scholes-Merton model for European options"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """
        Parameters:
        -----------
        S0: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        
    def d1(self) -> float:
        """Calculate d1 parameter"""
        return (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))
    
    def d2(self) -> float:
        """Calculate d2 parameter"""
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        """Calculate European call option price"""
        d1, d2 = self.d1(), self.d2()
        return self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - \
               self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self) -> float:
        """Calculate European put option price"""
        d1, d2 = self.d1(), self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - \
               self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
    
    def delta(self, option_type: str = 'call') -> float:
        """Calculate option delta"""
        if option_type.lower() == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self.d1())
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self.d1())
    
    def gamma(self) -> float:
        """Calculate option gamma"""
        return np.exp(-self.q * self.T) * norm.pdf(self.d1()) / \
               (self.S0 * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """Calculate option vega (per 1% change in volatility)"""
        return self.S0 * np.exp(-self.q * self.T) * norm.pdf(self.d1()) * np.sqrt(self.T) / 100
    
    def theta(self, option_type: str = 'call') -> float:
        """Calculate option theta (per day)"""
        d1, d2 = self.d1(), self.d2()
        term1 = -self.S0 * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        
        if option_type.lower() == 'call':
            term2 = self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1)
            term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            term2 = -self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return (term1 + term2 + term3) / 365
    
    def rho(self, option_type: str = 'call') -> float:
        """Calculate option rho (per 1% change in interest rate)"""
        d2 = self.d2()
        if option_type.lower() == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100


class HestonModel:
    """
    Heston Stochastic Volatility Model
    dS = r*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW2
    where dW1*dW2 = rho*dt
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, q: float,
                 v0: float, kappa: float, theta: float, sigma_v: float, rho: float):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma_v: Volatility of volatility
        rho: Correlation between stock and variance
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        
    def characteristic_function(self, phi: complex, j: int) -> complex:
        """Heston characteristic function"""
        if j == 1:
            u, b = 0.5, self.kappa - self.rho * self.sigma_v
        else:
            u, b = -0.5, self.kappa
            
        a = self.kappa * self.theta
        x = np.log(self.S0)
        
        d = np.sqrt((self.rho * self.sigma_v * phi * 1j - b)**2 - 
                   self.sigma_v**2 * (2 * u * phi * 1j - phi**2))
        
        g = (b - self.rho * self.sigma_v * phi * 1j + d) / \
            (b - self.rho * self.sigma_v * phi * 1j - d)
        
        C = (self.r - self.q) * phi * 1j * self.T + \
            a / self.sigma_v**2 * ((b - self.rho * self.sigma_v * phi * 1j + d) * self.T - 
            2 * np.log((1 - g * np.exp(d * self.T)) / (1 - g)))
        
        D = (b - self.rho * self.sigma_v * phi * 1j + d) / self.sigma_v**2 * \
            ((1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T)))
        
        return np.exp(C + D * self.v0 + 1j * phi * x)
    
    def call_price(self) -> float:
        """Calculate call price using Heston model via Fourier transform"""
        def integrand(phi, j):
            cf = self.characteristic_function(phi, j)
            return np.real(np.exp(-1j * phi * np.log(self.K)) * cf / (1j * phi))
        
        P1, _ = quad(lambda phi: integrand(phi, 1), 0, 100)
        P2, _ = quad(lambda phi: integrand(phi, 2), 0, 100)
        
        P1 = 0.5 + P1 / np.pi
        P2 = 0.5 + P2 / np.pi
        
        return self.S0 * np.exp(-self.q * self.T) * P1 - \
               self.K * np.exp(-self.r * self.T) * P2
    
    def put_price(self) -> float:
        """Calculate put price using put-call parity"""
        call = self.call_price()
        return call - self.S0 * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)


class MertonJumpDiffusion:
    """
    Merton Jump Diffusion Model
    dS = (r - lambda*k)*S*dt + sigma*S*dW + (J-1)*S*dN
    where J ~ lognormal(m, delta^2) and N is a Poisson process
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 lambda_: float, m: float, delta: float, q: float = 0.0):
        """
        Parameters:
        -----------
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Diffusion volatility
        lambda_: Jump intensity (jumps per year)
        m: Mean of log jump size
        delta: Std dev of log jump size
        q: Dividend yield
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lambda_ = lambda_
        self.m = m
        self.delta = delta
        self.q = q
        
    def call_price(self, n_terms: int = 50) -> float:
        """Calculate call price using series expansion"""
        k = np.exp(self.m + 0.5 * self.delta**2) - 1  # Expected jump size
        price = 0.0
        
        import math
        
        for n in range(n_terms):
            # Probability of n jumps
            prob_n = np.exp(-self.lambda_ * self.T) * (self.lambda_ * self.T)**n / math.factorial(n)
            
            # Adjusted parameters for n jumps
            sigma_n = np.sqrt(self.sigma**2 + n * self.delta**2 / self.T)
            r_n = self.r - self.lambda_ * k + n * (self.m + 0.5 * self.delta**2) / self.T
            
            # Black-Scholes price with adjusted parameters
            bs = BlackScholesModel(self.S0, self.K, self.T, r_n, sigma_n, self.q)
            price += prob_n * bs.call_price()
            
            # Check convergence
            if prob_n < 1e-10:
                break
                
        return price
    
    def put_price(self, n_terms: int = 50) -> float:
        """Calculate put price using series expansion"""
        k = np.exp(self.m + 0.5 * self.delta**2) - 1
        price = 0.0
        
        import math
        
        for n in range(n_terms):
            prob_n = np.exp(-self.lambda_ * self.T) * (self.lambda_ * self.T)**n / math.factorial(n)
            
            sigma_n = np.sqrt(self.sigma**2 + n * self.delta**2 / self.T)
            r_n = self.r - self.lambda_ * k + n * (self.m + 0.5 * self.delta**2) / self.T
            
            bs = BlackScholesModel(self.S0, self.K, self.T, r_n, sigma_n, self.q)
            price += prob_n * bs.put_price()
            
            if prob_n < 1e-10:
                break
                
        return price


class LocalVolatilityModel:
    """
    Dupire Local Volatility Model
    Computes local volatility surface from market option prices
    """
    
    def __init__(self, S0: float, r: float, q: float = 0.0):
        """
        Parameters:
        -----------
        S0: Current stock price
        r: Risk-free rate
        q: Dividend yield
        """
        self.S0 = S0
        self.r = r
        self.q = q
        
    def dupire_local_vol(self, K: float, T: float, 
                        market_call_price: callable,
                        dC_dK: callable, d2C_dK2: callable, dC_dT: callable) -> float:
        """
        Calculate local volatility using Dupire's formula
        
        sigma_local^2(K,T) = (dC/dT + q*C + (r-q)*K*dC/dK) / (0.5*K^2*d2C/dK2)
        """
        C = market_call_price(K, T)
        
        numerator = dC_dT(K, T) + self.q * C + (self.r - self.q) * K * dC_dK(K, T)
        denominator = 0.5 * K**2 * d2C_dK2(K, T)
        
        if denominator <= 0:
            return 0.0
            
        sigma_local_sq = numerator / denominator
        
        return np.sqrt(max(sigma_local_sq, 0.0))