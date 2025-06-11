from scipy.stats import norm
from scipy.optimize import fmin
from scipy.optimize import minimize
import numpy as np


def d1(S, K, r, T, iv, delta):
    return (np.log(S/K)+(r - delta + 0.5*iv**2)*T) / (iv*np.sqrt(T))

def d2(S, K, r, T, iv, delta):
    return (np.log(S/K)+(r -delta - 0.5*iv**2)*T) / (iv*np.sqrt(T))

def BS_Call(S, K, r, T, iv, delta):
    return S*np.exp(-delta*T)*norm.cdf(d1(S, K, r, T, iv, delta)) - K*np.exp(-r*T)*norm.cdf(d2(S, K, r, T, iv, delta))

def BS_Put(S, K, r, T, iv, delta):
    return K*np.exp(-r*T)*norm.cdf(-d2(S, K, r, T, iv, delta)) - S*np.exp(-delta*T)*norm.cdf(-d1(S, K, r, T, iv, delta))

# Call Greeks

def BS_Call_Delta(S, K, r, T, iv, delta):
    return np.exp(-delta*T)*norm.cdf(d1(S, K, r, T, iv, delta))

# Put Greeks

def BS_Put_Delta(S, K, r, T, iv, delta):
    return np.exp(-delta*T)*(norm.cdf(d1(S, K, r, T, iv, delta)) - 1)

def BS_Put_Gamma(S, K, r, T, iv, delta):
    return np.exp(-delta*T)*norm.pdf(d1(S, K, r, T, iv, delta)) / (S*iv*np.sqrt(T))

def BS_Put_Theta(S, K, r, T, iv, delta):
    part1 = -S*np.exp(-r*T)*r*norm.cdf(-d1(S, K, r, T, iv, delta))
    part2 = K*np.exp(-delta*T)*delta*norm.cdf(-d1(S, K, r, T, iv, delta) + iv*np.sqrt(T)) 
    part3 = -S*np.exp(-r*T)*iv*norm.pdf(d1(S, K, r, T, iv, delta)) / (2*np.sqrt(T))
    return (part1 + part2 + part3)



def InflectionPoint(S, K, r, T, delta):
    m = S / (K * np.exp(-(r-delta) * T))
    return np.sqrt(2 * np.abs(np.log(m)) / T)

def vega(S, K, r, T, sigma, delta):
    vega = K * np.exp(-r * T) * np.sqrt(T) * norm.pdf(d2(S, K, r, T, sigma, delta))
    return vega


def get_BS_IV(S, K, r, T, delta, marketPrice, optionType = 'call'):
    
    if optionType == 'C':
        f = lambda iv: (BS_Call(S, K, r, T, iv, delta) - marketPrice)**2
    elif optionType == 'P':
        f = lambda iv: (BS_Put(S, K, r, T, iv, delta) - marketPrice)**2
    else:
        Exception("Invalid option type")

    return fmin(f, 0.2, xtol=1e-12, ftol=1e-10, maxiter=1e6, maxfun=1e5, disp=False)[0]
    

def get_BS_IV_Newton(S, K, r, T, delta, marketPrice,  optionType = 'call', tol = 10e-4):

    x0 = InflectionPoint(S, K, r, T, delta)
    v = vega(S, K, r, T, x0, delta)

    if optionType == 'C':
        price = BS_Call(S, K, r, T, x0, delta)
        while (abs((price - marketPrice) / v) > tol):
            x0 = x0 - (price - marketPrice) / v
            price = BS_Call(S, K, r, T, x0, delta)
            v = vega(S, K, r, T, x0, delta)

    elif optionType == 'P':
        price = BS_Put(S, K, r, T, x0, delta)
        while (abs((price - marketPrice) / v) > tol):
            x0 = x0 - (price - marketPrice) / v
            price = BS_Put(S, K, r, T, x0, delta)
            v = vega(S, K, r, T, x0, delta)

    else:
        Exception("Invalid option type")

    return x0

