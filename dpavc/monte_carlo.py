"""
Advanced Monte Carlo Simulation with Variance Reduction Techniques
Implements standard MC, antithetic variates, control variates, and importance sampling
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results"""
    price: float
    std_error: float
    confidence_interval: Tuple[float, float]
    paths_used: int
    variance_reduction: Optional[str] = None
    
    def __str__(self):
        return f"""
Price: ${self.price:.6f}
Std Error: ${self.std_error:.6f}
95% CI: [{self.confidence_interval[0]:.6f}, {self.confidence_interval[1]:.6f}]
Paths: {self.paths_used:,}
Variance Reduction: {self.variance_reduction or 'None'}
"""


class MonteCarloEngine:
    """Advanced Monte Carlo pricing engine with multiple variance reduction techniques"""
    
    def __init__(self, n_simulations: int = 100000, seed: Optional[int] = 42):
        """
        Parameters:
        -----------
        n_simulations: Number of Monte Carlo paths
        seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def standard_monte_carlo(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, q: float = 0.0, 
                           option_type: str = 'call') -> SimulationResult:
        """Standard Monte Carlo simulation without variance reduction"""
        
        # Generate random samples
        Z = np.random.standard_normal(self.n_simulations)
        
        # Simulate terminal stock prices
        ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        # 95% confidence interval
        ci = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        return SimulationResult(price, std_error, ci, self.n_simulations, "None")
    
    def antithetic_variates(self, S0: float, K: float, T: float, r: float,
                           sigma: float, q: float = 0.0,
                           option_type: str = 'call') -> SimulationResult:
        """
        Monte Carlo with antithetic variates
        Uses both Z and -Z to reduce variance
        """
        
        n_half = self.n_simulations // 2
        Z = np.random.standard_normal(n_half)
        
        # Original paths
        ST1 = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Antithetic paths
        ST2 = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-Z))
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs1 = np.maximum(ST1 - K, 0)
            payoffs2 = np.maximum(ST2 - K, 0)
        else:
            payoffs1 = np.maximum(K - ST1, 0)
            payoffs2 = np.maximum(K - ST2, 0)
        
        # Average antithetic pairs
        payoffs = (payoffs1 + payoffs2) / 2
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_half)
        
        ci = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        return SimulationResult(price, std_error, ci, self.n_simulations, "Antithetic Variates")
    
    def control_variates(self, S0: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0.0,
                        option_type: str = 'call') -> SimulationResult:
        """
        Monte Carlo with control variates
        Uses the stock price as a control variate (known expectation)
        """
        
        Z = np.random.standard_normal(self.n_simulations)
        ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Use stock price as control variate
        # E[ST] = S0 * exp((r-q)*T)
        control = ST
        expected_control = S0 * np.exp((r - q) * T)
        
        # Calculate optimal beta coefficient
        cov = np.cov(payoffs, control)[0, 1]
        var_control = np.var(control)
        beta = cov / var_control if var_control > 0 else 0
        
        # Adjust payoffs using control variate
        adjusted_payoffs = payoffs - beta * (control - expected_control)
        
        price = np.exp(-r * T) * np.mean(adjusted_payoffs)
        std_error = np.exp(-r * T) * np.std(adjusted_payoffs) / np.sqrt(self.n_simulations)
        
        ci = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        return SimulationResult(price, std_error, ci, self.n_simulations, "Control Variates")
    
    def heston_monte_carlo(self, S0: float, K: float, T: float, r: float, q: float,
                          v0: float, kappa: float, theta: float, 
                          sigma_v: float, rho: float,
                          n_steps: int = 252,
                          option_type: str = 'call') -> SimulationResult:
        """
        Monte Carlo simulation for Heston model
        Uses Euler discretization with full truncation scheme
        """
        
        dt = T / n_steps
        
        # Generate correlated random numbers
        Z1 = np.random.standard_normal((self.n_simulations, n_steps))
        Z2_indep = np.random.standard_normal((self.n_simulations, n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2_indep
        
        # Initialize arrays
        S = np.zeros((self.n_simulations, n_steps + 1))
        v = np.zeros((self.n_simulations, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Simulate paths
        for i in range(n_steps):
            # Full truncation scheme for variance
            v_pos = np.maximum(v[:, i], 0)
            
            # Update variance (CIR process)
            v[:, i + 1] = v[:, i] + kappa * (theta - v_pos) * dt + \
                         sigma_v * np.sqrt(v_pos * dt) * Z2[:, i]
            
            # Update stock price
            S[:, i + 1] = S[:, i] * np.exp((r - q - 0.5 * v_pos) * dt + \
                                          np.sqrt(v_pos * dt) * Z1[:, i])
        
        # Calculate payoffs
        ST = S[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        ci = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        return SimulationResult(price, std_error, ci, self.n_simulations, "Heston MC")
    
    def jump_diffusion_monte_carlo(self, S0: float, K: float, T: float, r: float,
                                   sigma: float, lambda_: float, m: float, delta: float,
                                   q: float = 0.0, n_steps: int = 252,
                                   option_type: str = 'call') -> SimulationResult:
        """
        Monte Carlo for Merton Jump Diffusion model
        """
        
        dt = T / n_steps
        
        # Simulate paths
        S = np.zeros((self.n_simulations, n_steps + 1))
        S[:, 0] = S0
        
        for i in range(n_steps):
            # Diffusion component
            Z = np.random.standard_normal(self.n_simulations)
            diffusion = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            
            # Jump component (Poisson process)
            N_jumps = np.random.poisson(lambda_ * dt, self.n_simulations)
            
            # Jump sizes (log-normal)
            jump_component = np.zeros(self.n_simulations)
            for j in range(self.n_simulations):
                if N_jumps[j] > 0:
                    jumps = np.random.normal(m, delta, N_jumps[j])
                    jump_component[j] = np.sum(jumps)
            
            # Update stock price
            S[:, i + 1] = S[:, i] * np.exp(diffusion + jump_component)
        
        # Calculate payoffs
        ST = S[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        ci = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        return SimulationResult(price, std_error, ci, self.n_simulations, "Jump Diffusion MC")
    
    def compare_variance_reduction(self, S0: float, K: float, T: float, r: float,
                                   sigma: float, q: float = 0.0,
                                   option_type: str = 'call') -> Dict[str, SimulationResult]:
        """
        Compare different variance reduction techniques
        Returns results for all methods
        """
        
        results = {
            'Standard': self.standard_monte_carlo(S0, K, T, r, sigma, q, option_type),
            'Antithetic': self.antithetic_variates(S0, K, T, r, sigma, q, option_type),
            'Control Variates': self.control_variates(S0, K, T, r, sigma, q, option_type)
        }
        
        return results
    
    def variance_reduction_efficiency(self, results: Dict[str, SimulationResult]) -> Dict[str, float]:
        """
        Calculate efficiency gain from variance reduction
        Efficiency = (Var_standard / Var_method)
        """
        
        if 'Standard' not in results:
            raise ValueError("Need standard MC results for comparison")
        
        std_variance = results['Standard'].std_error ** 2
        
        efficiency = {}
        for method, result in results.items():
            if method != 'Standard':
                method_variance = result.std_error ** 2
                efficiency[method] = std_variance / method_variance if method_variance > 0 else 0
        
        return efficiency