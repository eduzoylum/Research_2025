import numpy as np
import pandas as pd
from BS_utils import bs_functions as bf
from scipy.optimize import fmin
import matplotlib.pyplot as plt


def get_dividend_curve(option_data, type = 'smooth', **kwargs):
    if type == 'smooth':
        return DividendCurve_smooth(option_data, **kwargs)
    elif type == 'step':
        return DividendCurve_step(option_data, **kwargs)
    else:
        raise(ValueError('Type should be either smooth or step.'))

def get_implied_dividend(options_data, S_0, **kwargs):

    maturities = options_data['T'].unique()

    for t in maturities:

        maturityData = options_data[ options_data['T'] == t ]
        q_implied = _get_implied_q_for_maturity(maturityData, S_0, **kwargs)

        options_data.loc[ options_data['T'] == t,'q'] =  np.repeat(q_implied, maturityData.shape[0])

    return options_data['q'].values

def compute_implied_dividends_from_quotes(option_data, S_0):
    """
    Compute implied dividend yields from call-put parity using  /ask prices.
    
    Parameters:
    - df: DataFrame containing both calls and puts with:
        ['option_type', 'strike', 'bid', 'ask', 'T', 'r']
    - S_0: Spot price (float)

    Returns:
    - DataFrame with matched call/put pairs and implied dividend yields:
        ['strike', 'T', 'r', 'delta_long_call_short_put', 'delta_short_call_long_put']
    """
    
    maturities = option_data['T'].unique()

    for maturity in maturities:

        df = option_data[ option_data['T'] == maturity ]
        df = df[ df['open_interest'] > 50 ]
        df = df[ df['bid_1545'] > 0 ]
        df = df[ df['ask_1545'] > 0 ]

        # Separate calls and puts
        calls = df[df['option_type'] == 'C'].copy()
        puts  = df[df['option_type'] == 'P'].copy()

        # Merge on strike, T, r
        merged = pd.merge(
            calls,
            puts,
            on=['strike','T','r'],
            suffixes=('_call', '_put')
        )

        if merged.shape[0] == 0:
            option_data.loc[ option_data['T'] == maturity , 'q' ] = np.repeat(np.nan ,option_data.loc[ option_data['T'] == maturity].shape[0])            
            print("No valid call-put pairs after filtering for maturity "+str(maturity)+'.')
            continue

        # Pull required values
        K  = merged['strike']
        T  = merged['T']
        r  = merged['r']
        openInt = merged['open_interest_call'] + merged['open_interest_put']
        merged['openInterest'] = openInt
        C_ask = merged['ask_1545_call']
        C_bid = merged['bid_1545_call']
        P_ask = merged['ask_1545_put']
        P_bid = merged['bid_1545_put']

        def compute_delta(C, P, K, T, r):
            inside = (C - P + K * np.exp(-r * T)) / S_0
            inside = np.clip(inside, 1e-10, None)  # avoid log blow-up
            return -np.log(inside) / T
        
        # Long call (ask), short put (bid)
        merged['delta_long_call_short_put'] = compute_delta(C_ask, P_bid, K, T, r)

        # Short call (bid), long put (ask)
        merged['delta_short_call_long_put'] = compute_delta(C_bid, P_ask, K, T, r)

        merged['combined_dividend'] = (merged['delta_long_call_short_put']*merged['open_interest_put'] + merged['delta_short_call_long_put']*merged['open_interest_call'])/(merged['open_interest_call']+merged['open_interest_put'])

        q_final = merged['combined_dividend']@(merged['open_interest_call']+merged['open_interest_put'])/(merged['open_interest_call']+merged['open_interest_put']).sum()

        option_data.loc[ option_data['T'] == maturity , 'q' ] = np.repeat(q_final,option_data.loc[ option_data['T'] == maturity].shape[0])

    return option_data['q'].values

def _get_implied_q_for_maturity(maturityData, S_0, **kwargs):

    maturityData = maturityData[maturityData['openInterest'] > 75]

    if maturityData.shape[0] == 0:
        return np.nan

    if 'iv_label' in kwargs:
        iv_label = kwargs['iv_label']
    else:
        iv_label = 'impliedVolatility'

    # Get the known parameters of the Black-Scholes model
    K = maturityData['strike'].values
    r = maturityData['r'].values
    T = maturityData['T'].values
    Sigma = maturityData[iv_label].values

    # Get the option type
    OptionType = maturityData['option_type'].values

    # Get the mid prices and open interest
    midPrices = ((maturityData['bid_1545'] + maturityData['ask_1545']) / 2).values
    volume = maturityData['openInterest'].values
    volume = volume / np.sum(volume)

    def f(params):

        q = params

        fittedPrices = []
        for i in range(maturityData.shape[0]):
            if OptionType[i] == 'C':
                price = bf.BS_Call(S_0, K[i], r[i], T[i], Sigma[i], q)
            else:
                price = bf.BS_Put(S_0, K[i], r[i], T[i], Sigma[i], q)
            
            fittedPrices.append(price)

        # Use the open interest as the weight for the error

        return np.mean(volume*(fittedPrices - midPrices)**2)

    
    return  fmin(f , 0.01,xtol=1e-5, ftol=1e-10, maxiter=1e5, maxfun=1e5, disp=False)


class DividendCurve_smooth:
    def __init__(self, T_q_df, **kwargs):
        """
        T_q_df: DataFrame with columns:
            - 'T': maturities in years
            - 'q_imp': implied dividend yield (in decimals)

        Handles duplicates by averaging yields at the same maturity.
        """
        if 'q_imp_ind' in kwargs:
            q_imp_ind = kwargs['q_imp_ind']
        else:
            q_imp_ind = 'q_imp'

            
        # Collapse multiple rows with same T by averaging q_imp
        cleaned_df = (
            T_q_df.groupby("T", as_index=False)
            .agg({q_imp_ind: "mean"})
            .sort_values("T")
        ).dropna()

        self.T = np.array(cleaned_df["T"].values)
        self.q = np.array(cleaned_df[q_imp_ind].values)
        self.integrals = self._precompute_integrals()

    def _precompute_integrals(self):
        """
        Precompute cumulative integral I(T_i) = sum q_j * (T_j - T_{j-1})
        """
        I = np.zeros_like(self.T)
        for i in range(1, len(self.T)):
            delta = self.T[i] - self.T[i - 1]
            I[i] = I[i - 1] + self.q[i - 1] * delta
        return I

    def appreciation(self, t):
        """
        Compute A(t) = exp(-∫₀ᵗ q(s) ds), fast with precomputed logic
        """
        if t <= 0:
            return 1.0

        idx = np.searchsorted(self.T, t, side='right') - 1
        idx = max(idx, 0)

        integral_to_tk = self.integrals[idx]
        tail = t - self.T[idx] if idx < len(self.T) else 0.0
        q_t = self.q[idx] if idx < len(self.q) else self.q[-1]

        total_integral = integral_to_tk + q_t * tail
        return np.exp(-total_integral)

    def q_at(self, t):
        """
        Effective dividend yield implied by A(t)
        """
        if t <= 0:
            return 0.0
        A = self.appreciation(t)
        return -np.log(A) / t

    def plot(self, t_max=6, num_points=300):
        """
        Plot appreciation factor and implied dividend yield curve
        """
        T_grid = np.linspace(0, t_max, num_points)
        A_vals = np.array([self.appreciation(t) for t in T_grid])
        q_vals = np.array([self.q_at(t) for t in T_grid])
        A_breaks = np.array([self.appreciation(t) for t in self.T])

        plt.figure(figsize=(12, 6))

        # Plot A(t)
        plt.subplot(1, 2, 1)
        plt.plot(T_grid, A_vals, label="A(t)", color = 'black')
        plt.scatter(self.T, A_breaks, color='red', label="Breakpoints", zorder=5, s = 3)
        plt.title("Stock Appreciation Factor")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("A(t)")
        plt.grid(True)
        plt.legend()

        # Plot q(t)
        plt.subplot(1, 2, 2)
        plt.plot(T_grid, q_vals, label="Implied q(t)", color = 'black')
        plt.scatter(self.T, self.q, color='red', zorder=5, s =3)
        plt.title("Dividend Yield Curve")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("q(t)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


class DividendCurve_step:
    def __init__(self, T_q_df, **kwargs):
        """
        T_q_df: DataFrame with columns:
            - 'T': maturities in years
            - 'q_imp_ind': column name for implied dividend yield (in decimals)
        """

        if 'q_imp_ind' in kwargs:
            q_imp_ind = kwargs['q_imp_ind']
        else:
            q_imp_ind = 'q_imp'

        # Average across repeated maturities if needed
        cleaned_df = (
            T_q_df.groupby("T", as_index=False)
            .agg({q_imp_ind: "mean"})
            .sort_values("T")
        )
        cleaned_df = cleaned_df[['T',q_imp_ind]].dropna()

        self.T = np.array(cleaned_df["T"].values)
        self.q = np.array(cleaned_df[q_imp_ind].values)

        # Precompute A(T_k) = exp(-q_k * T_k) for all breakpoints
        self.appreciation_map = {
            t: np.exp(-q * t) for t, q in zip(self.T, self.q)
        }

    def appreciation(self, t):
        """
        Compute A(t) = A(T_k) * exp(-q_k * (t - T_k))
        where T_k is the latest breakpoint before t.
        """
        if t <= 0:
            return 1.0

        idx = np.searchsorted(self.T, t, side='right') - 1
        idx = max(idx, 0)

        T_k = self.T[idx]
        q_k = self.q[idx]
        A_k = self.appreciation_map[T_k]

        return A_k * np.exp(-q_k * (t - T_k))

    def q_at(self, t):
        """
        Effective dividend yield implied by A(t)
        """
        if t <= 0:
            return 0.0
        A = self.appreciation(t)
        return -np.log(A) / t

    def plot(self, t_max=6.0, num_points=300):
        """
        Plot appreciation factor and implied dividend yield curve
        """
        T_grid = np.linspace(0, t_max, num_points)
        A_vals = np.array([self.appreciation(t) for t in T_grid])
        q_vals = np.array([self.q_at(t) for t in T_grid])
        A_breaks = np.array([self.appreciation(t) for t in self.T])

        plt.figure(figsize=(12, 6))

        # Plot A(t)
        plt.subplot(1, 2, 1)
        plt.plot(T_grid, A_vals, label="A(t)",color='black')
        plt.scatter(self.T, A_breaks, color='red', label="Breakpoints", zorder=5, s=3)
        plt.title("Stock Appreciation Factor")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("A(t)")
        plt.grid(True)
        plt.legend()

        # Plot q(t)
        plt.subplot(1, 2, 2)
        plt.plot(T_grid, q_vals, label="Implied q(t)", color= 'black')
        plt.scatter(self.T, self.q, color='red', zorder=5, s=3)
        plt.title("Dividend Yield Curve")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("q(t)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()