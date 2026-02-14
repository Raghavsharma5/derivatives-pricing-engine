"""
Market Data Calibration
Calibrate models to real market option prices and volatility surfaces
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
import warnings

from dpavc.models import HestonModel, MertonJumpDiffusion
from dpavc.imp_vol import ImpliedVolatility


class ModelCalibrator:
    """Calibrate option pricing models to market data"""
    
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
        
    def calibrate_heston(self, market_data: pd.DataFrame,
                        method: str = 'differential_evolution') -> Dict[str, float]:
        """
        Calibrate Heston model to market option prices
        
        Parameters:
        -----------
        market_data: DataFrame with columns ['strike', 'maturity', 'market_price', 'option_type']
        method: 'differential_evolution' (global) or 'local' (local optimization)
        
        Returns:
        --------
        Dictionary with calibrated Heston parameters
        """
        
        def objective(params):
            v0, kappa, theta, sigma_v, rho = params
            
            # Parameter constraints to ensure valid model
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0:
                return 1e10
            if abs(rho) >= 1:
                return 1e10
            # Feller condition: 2*kappa*theta > sigma_v^2
            if 2 * kappa * theta <= sigma_v**2:
                return 1e10
            
            total_error = 0.0
            
            for _, row in market_data.iterrows():
                try:
                    heston = HestonModel(
                        self.S0, row['strike'], row['maturity'],
                        self.r, self.q, v0, kappa, theta, sigma_v, rho
                    )
                    
                    if row['option_type'].lower() == 'call':
                        model_price = heston.call_price()
                    else:
                        model_price = heston.put_price()
                    
                    # Weighted squared error (weight by vega)
                    error = (model_price - row['market_price'])**2
                    total_error += error
                    
                except Exception:
                    return 1e10
            
            return total_error / len(market_data)
        
        if method == 'differential_evolution':
            # Global optimization - better for Heston due to multiple local minima
            bounds = [
                (0.01, 1.0),    # v0
                (0.1, 10.0),    # kappa
                (0.01, 1.0),    # theta
                (0.01, 2.0),    # sigma_v
                (-0.99, 0.99)   # rho
            ]
            
            result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
            
        else:
            # Local optimization
            initial_guess = [0.04, 2.0, 0.04, 0.3, -0.5]
            bounds = [
                (0.01, 1.0),
                (0.1, 10.0),
                (0.01, 1.0),
                (0.01, 2.0),
                (-0.99, 0.99)
            ]
            
            result = minimize(objective, initial_guess, method='L-BFGS-B', 
                            bounds=bounds)
        
        if result.success or method == 'differential_evolution':
            return {
                'v0': result.x[0],
                'kappa': result.x[1],
                'theta': result.x[2],
                'sigma_v': result.x[3],
                'rho': result.x[4],
                'rmse': np.sqrt(result.fun)
            }
        else:
            raise ValueError("Heston calibration failed")
    
    def calibrate_jump_diffusion(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calibrate Merton Jump Diffusion model to market data
        
        Parameters:
        -----------
        market_data: DataFrame with columns ['strike', 'maturity', 'market_price', 'option_type']
        
        Returns:
        --------
        Dictionary with calibrated jump diffusion parameters
        """
        
        def objective(params):
            sigma, lambda_, m, delta = params
            
            if sigma <= 0 or lambda_ < 0 or delta <= 0:
                return 1e10
            
            total_error = 0.0
            
            for _, row in market_data.iterrows():
                try:
                    jd = MertonJumpDiffusion(
                        self.S0, row['strike'], row['maturity'],
                        self.r, sigma, lambda_, m, delta, self.q
                    )
                    
                    if row['option_type'].lower() == 'call':
                        model_price = jd.call_price()
                    else:
                        model_price = jd.put_price()
                    
                    error = (model_price - row['market_price'])**2
                    total_error += error
                    
                except Exception:
                    return 1e10
            
            return total_error / len(market_data)
        
        # Global optimization
        bounds = [
            (0.01, 1.0),   # sigma
            (0.0, 5.0),    # lambda
            (-0.5, 0.5),   # m
            (0.01, 0.5)    # delta
        ]
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
        
        return {
            'sigma': result.x[0],
            'lambda': result.x[1],
            'm': result.x[2],
            'delta': result.x[3],
            'rmse': np.sqrt(result.fun)
        }
    
    def model_comparison(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare calibration quality of different models
        
        Returns:
        --------
        DataFrame with RMSE and other metrics for each model
        """
        
        results = {}
        
        try:
            # Calibrate Heston
            heston_params = self.calibrate_heston(market_data)
            results['Heston'] = {
                'RMSE': heston_params['rmse'],
                'Parameters': {k: v for k, v in heston_params.items() if k != 'rmse'}
            }
        except Exception as e:
            warnings.warn(f"Heston calibration failed: {str(e)}")
        
        try:
            # Calibrate Jump Diffusion
            jd_params = self.calibrate_jump_diffusion(market_data)
            results['Jump Diffusion'] = {
                'RMSE': jd_params['rmse'],
                'Parameters': {k: v for k, v in jd_params.items() if k != 'rmse'}
            }
        except Exception as e:
            warnings.warn(f"Jump Diffusion calibration failed: {str(e)}")
        
        # Convert to DataFrame
        comparison = pd.DataFrame([
            {'Model': model, 'RMSE': data['RMSE'], 
             'Num_Params': len(data['Parameters'])}
            for model, data in results.items()
        ])
        
        return comparison.sort_values('RMSE')


class MarketDataLoader:
    """Load and process market option data"""
    
    @staticmethod
    def generate_synthetic_surface(S0: float, strikes: np.ndarray, 
                                   maturities: np.ndarray,
                                   r: float = 0.05, q: float = 0.02,
                                   vol_atm: float = 0.25,
                                   skew: float = -0.15,
                                   smile: float = 0.05) -> pd.DataFrame:
        """
        Generate synthetic option surface for testing
        Uses parametric volatility smile
        
        Parameters:
        -----------
        vol_atm: ATM volatility
        skew: Slope of volatility skew
        smile: Curvature of volatility smile
        """
        
        data = []
        
        for T in maturities:
            F = S0 * np.exp((r - q) * T)  # Forward price
            
            for K in strikes:
                # Log-moneyness
                k = np.log(K / F)
                
                # Parametric volatility smile
                vol = vol_atm + skew * k + smile * k**2
                vol = max(vol, 0.05)  # Floor at 5%
                
                # Calculate option prices
                from dpavc.models import BlackScholesModel
                
                bs_call = BlackScholesModel(S0, K, T, r, vol, q)
                bs_put = BlackScholesModel(S0, K, T, r, vol, q)
                
                data.append({
                    'strike': K,
                    'maturity': T,
                    'market_price': bs_call.call_price(),
                    'option_type': 'call',
                    'implied_vol': vol
                })
                
                data.append({
                    'strike': K,
                    'maturity': T,
                    'market_price': bs_put.put_price(),
                    'option_type': 'put',
                    'implied_vol': vol
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def load_from_csv(filepath: str) -> pd.DataFrame:
        """
        Load market data from CSV file
        Expected columns: strike, maturity, market_price, option_type
        """
        
        df = pd.read_csv(filepath)
        
        required_cols = ['strike', 'maturity', 'market_price', 'option_type']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        return df
    
    @staticmethod
    def compute_implied_vol_surface(data: pd.DataFrame, S0: float, 
                                    r: float, q: float = 0.0) -> pd.DataFrame:
        """
        Compute implied volatility for all options in dataset
        """
        
        iv_data = data.copy()
        iv_data['implied_vol'] = 0.0
        
        for idx, row in iv_data.iterrows():
            try:
                iv_calc = ImpliedVolatility(S0, row['strike'], row['maturity'], r, q)
                iv, _ = iv_calc.newton_raphson(row['market_price'], row['option_type'])
                iv_data.at[idx, 'implied_vol'] = iv
            except Exception as e:
                warnings.warn(f"IV calculation failed for row {idx}: {str(e)}")
                iv_data.at[idx, 'implied_vol'] = np.nan
        
        return iv_data