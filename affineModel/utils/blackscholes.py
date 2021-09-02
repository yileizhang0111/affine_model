"""
BS formula for benchmark test and calculate implied vol
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def call(s, k, vol, tau, r, q):
    var = vol * np.sqrt(tau)
    if (s < 10e-6) or (var < 10e-6):
        return max(s * np.exp(-q * tau) - k * np.exp(-r * tau), 0.0)
    forward = s * np.exp((r - q) * tau)
    df = np.exp(-r * tau)
    if k < 0:
        ln_moneyness = 40.0
    else:
        ln_moneyness = np.log(forward / k)
    dp = ln_moneyness / var + 0.5 * var
    dm = dp - var
    return df * (forward * norm.cdf(dp) - k * norm.cdf(dm))


def put(s, k, vol, tau, r, q):
    var = vol * np.sqrt(tau)
    if (s < 10e-6) or (var < 10e-6):
        return max(k * np.exp(-r * tau) - s * np.exp(-q * tau), 0.0)
    forward = s * np.exp((r - q) * tau)
    df = np.exp(-r * tau)
    ln_moneyness = np.log(forward / k)
    dp = ln_moneyness / var + 0.5 * var
    dm = dp - var
    return df * (k * norm.cdf(-dm) - forward * norm.cdf(-dp))


def implvol_call(price, s, k, tau, r, q):
    return brentq(lambda x: price - call(s, k, x, tau, r, q), 0, 2)


def implvol_put(price, s, k, tau, r, q):
    return brentq(lambda x: price - put(s, k, x, tau, r, q), 0, 2)
