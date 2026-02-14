"""
Performance Optimization and Benchmarking
Compare NumPy, Numba, and vectorized implementations
"""

import numpy as np
import time
from typing import Dict, Callable
import pandas as pd
from functools import wraps

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")


def timer(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


class PerformanceBenchmark:
    """Benchmark different implementation strategies"""
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def black_scholes_python_loop(S0: np.ndarray, K: float, T: float, 
                                   r: float, sigma: float, q: float = 0.0) -> np.ndarray:
        """Pure Python implementation with loop - baseline (slowest)"""
        from scipy.stats import norm
        
        n = len(S0)
        prices = np.zeros(n)
        
        for i in range(n):
            d1 = (np.log(S0[i] / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            prices[i] = S0[i] * np.exp(-q * T) * norm.cdf(d1) - \
                       K * np.exp(-r * T) * norm.cdf(d2)
        
        return prices
    
    @staticmethod
    def black_scholes_numpy(S0: np.ndarray, K: float, T: float,
                           r: float, sigma: float, q: float = 0.0) -> np.ndarray:
        """Vectorized NumPy implementation (fast)"""
        from scipy.stats import norm
        
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    if NUMBA_AVAILABLE:
        @staticmethod
        @jit(nopython=True, parallel=True)
        def black_scholes_numba(S0: np.ndarray, K: float, T: float,
                               r: float, sigma: float, q: float = 0.0) -> np.ndarray:
            """Numba JIT-compiled implementation with parallelization (fastest)"""
            
            n = len(S0)
            prices = np.zeros(n)
            
            # Standard normal CDF approximation (since scipy not available in nopython mode)
            def norm_cdf(x):
                return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
            
            for i in prange(n):
                d1 = (np.log(S0[i] / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                prices[i] = S0[i] * np.exp(-q * T) * norm_cdf(d1) - \
                           K * np.exp(-r * T) * norm_cdf(d2)
            
            return prices
    
    @staticmethod
    @timer
    def benchmark_black_scholes(n_prices: int = 1000000, method: str = 'numpy') -> np.ndarray:
        """Benchmark Black-Scholes calculation"""
        
        # Generate random spot prices
        S0 = np.random.uniform(80, 120, n_prices)
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.25
        q = 0.02
        
        if method == 'python_loop':
            return PerformanceBenchmark.black_scholes_python_loop(S0, K, T, r, sigma, q)
        elif method == 'numpy':
            return PerformanceBenchmark.black_scholes_numpy(S0, K, T, r, sigma, q)
        elif method == 'numba' and NUMBA_AVAILABLE:
            return PerformanceBenchmark.black_scholes_numba(S0, K, T, r, sigma, q)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    @timer
    def benchmark_monte_carlo(n_simulations: int = 100000, 
                             n_steps: int = 252,
                             method: str = 'numpy') -> float:
        """Benchmark Monte Carlo simulation"""
        
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.25
        q = 0.02
        
        dt = T / n_steps
        
        if method == 'python_loop':
            # Pure Python implementation
            prices = []
            for _ in range(n_simulations):
                S = S0
                for _ in range(n_steps):
                    Z = np.random.standard_normal()
                    S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
                prices.append(max(S - K, 0))
            
            return np.exp(-r * T) * np.mean(prices)
        
        elif method == 'numpy':
            # Vectorized NumPy
            Z = np.random.standard_normal((n_simulations, n_steps))
            log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            S_paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
            ST = S_paths[:, -1]
            payoffs = np.maximum(ST - K, 0)
            
            return np.exp(-r * T) * np.mean(payoffs)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive performance benchmarks"""
        
        results = []
        
        # Black-Scholes benchmarks
        print("Benchmarking Black-Scholes implementations...")
        
        for n_prices in [10000, 100000, 1000000]:
            # Python loop
            _, time_python = self.benchmark_black_scholes(n_prices, 'python_loop')
            results.append({
                'Task': 'Black-Scholes',
                'Method': 'Python Loop',
                'Size': n_prices,
                'Time (s)': time_python,
                'Speedup': 1.0
            })
            
            # NumPy
            _, time_numpy = self.benchmark_black_scholes(n_prices, 'numpy')
            results.append({
                'Task': 'Black-Scholes',
                'Method': 'NumPy',
                'Size': n_prices,
                'Time (s)': time_numpy,
                'Speedup': time_python / time_numpy
            })
            
            # Numba (if available)
            if NUMBA_AVAILABLE:
                # Warm-up JIT
                _ = self.benchmark_black_scholes(1000, 'numba')
                
                _, time_numba = self.benchmark_black_scholes(n_prices, 'numba')
                results.append({
                    'Task': 'Black-Scholes',
                    'Method': 'Numba',
                    'Size': n_prices,
                    'Time (s)': time_numba,
                    'Speedup': time_python / time_numba
                })
        
        # Monte Carlo benchmarks
        print("Benchmarking Monte Carlo implementations...")
        
        for n_sims in [10000, 50000, 100000]:
            # Python loop
            _, time_python = self.benchmark_monte_carlo(n_sims, 252, 'python_loop')
            results.append({
                'Task': 'Monte Carlo',
                'Method': 'Python Loop',
                'Size': n_sims,
                'Time (s)': time_python,
                'Speedup': 1.0
            })
            
            # NumPy
            _, time_numpy = self.benchmark_monte_carlo(n_sims, 252, 'numpy')
            results.append({
                'Task': 'Monte Carlo',
                'Method': 'NumPy',
                'Size': n_sims,
                'Time (s)': time_numpy,
                'Speedup': time_python / time_numpy
            })
        
        df = pd.DataFrame(results)
        self.results = df
        return df
    
    def generate_report(self) -> str:
        """Generate performance benchmark report"""
        
        if self.results is None or len(self.results) == 0:
            return "No benchmark results available. Run run_comprehensive_benchmark() first."
        
        report = ["=" * 80]
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary by method
        report.append("Average Speedup by Method:")
        report.append("-" * 80)
        
        summary = self.results.groupby('Method')['Speedup'].agg(['mean', 'min', 'max'])
        report.append(summary.to_string())
        report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 80)
        report.append(self.results.to_string(index=False))
        report.append("")
        
        # Key findings
        report.append("Key Findings:")
        report.append("-" * 80)
        
        best_method = self.results.loc[self.results['Speedup'].idxmax()]
        report.append(f"• Best performance: {best_method['Method']} for {best_method['Task']}")
        report.append(f"  ({best_method['Speedup']:.1f}x speedup over Python loop)")
        
        numpy_speedup = self.results[self.results['Method'] == 'NumPy']['Speedup'].mean()
        report.append(f"• NumPy average speedup: {numpy_speedup:.1f}x")
        
        if NUMBA_AVAILABLE:
            numba_speedup = self.results[self.results['Method'] == 'Numba']['Speedup'].mean()
            report.append(f"• Numba average speedup: {numba_speedup:.1f}x")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


class MemoryProfiler:
    """Profile memory usage of different implementations"""
    
    @staticmethod
    def estimate_memory_usage(n_simulations: int, n_steps: int, precision: str = 'float64') -> Dict[str, float]:
        """
        Estimate memory usage for Monte Carlo simulation
        
        Returns memory in MB
        """
        
        bytes_per_element = 8 if precision == 'float64' else 4
        
        # Path storage: n_simulations × n_steps
        paths_memory = (n_simulations * n_steps * bytes_per_element) / (1024**2)
        
        # Final prices: n_simulations
        final_memory = (n_simulations * bytes_per_element) / (1024**2)
        
        # Random numbers: n_simulations × n_steps
        random_memory = (n_simulations * n_steps * bytes_per_element) / (1024**2)
        
        return {
            'paths': paths_memory,
            'final_prices': final_memory,
            'random_numbers': random_memory,
            'total': paths_memory + final_memory + random_memory
        }