#!/usr/bin/env python3
from dpavc.models import BlackScholesModel
import numpy as np

# Quick test
bs = BlackScholesModel(S0=100, K=100, T=0.25, r=0.05, sigma=0.25, q=0.02)
print(f'Call Price: ${bs.call_price():.6f}')
print(f'Put Price: ${bs.put_price():.6f}')
print(f'Delta: {bs.delta("call"):.6f}')
print(f'Gamma: {bs.gamma():.6f}')
print(f'Vega: {bs.vega():.6f}')
print('BSM check!')