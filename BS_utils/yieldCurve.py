import datetime
import numpy as np
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt


# Series IDs for zero-coupon (spot) Treasury yields
FRED_TREASURY_SERIES = {
    1/12: "DGS1MO",
    3/12: "DGS3MO",
    6/12: "DGS6MO",
    1:    "DGS1",
    2:    "DGS2",
    3:    "DGS3",
    5:    "DGS5",
    7:    "DGS7",
    10:   "DGS10",
    20:   "DGS20",
    30:   "DGS30"
}


def fetch_hybrid_yield_curve(ref_date, fred_client = None):
    """
    Fetches yield curve:
    - DGS* for short maturities (1M, 3M)
    - BC_* zero-coupon rates from 6M onward
    Returns dict { maturity in years: yield as decimal }
    """
    if fred_client is None:
        # Return an error if no Fred client is provided
        raise ValueError("A Fred client must be provided.")

    ref_date = pd.to_datetime(ref_date or datetime.date.today())

    curve = {}
    for maturity, series_id in FRED_TREASURY_SERIES.items():
        try:
            series = fred_client.get_series_latest_release(series_id).dropna()
            latest_rate = series[series.index <= ref_date].iloc[-1]
            curve[maturity] = latest_rate / 100.0  # Convert % to decimal
        except Exception as e:
            print(f"Warning: Could not fetch {series_id} — {e}")
    
    return curve


class YieldCurve:
    def __init__(self, ref_date = None, fred_client = None):
        """
        yield_dict: { maturity in years (float): yield (as decimal) }
        """
        if fred_client is None:
            # Initialize FRED client with your API key
            self.fred_client = Fred(api_key='fac7604001fe45ef764f2d068238b855')
        else:
            self.fred_client = fred_client

        self.yield_dict = fetch_hybrid_yield_curve(ref_date, self.fred_client)
        self.maturities = np.array(sorted(self.yield_dict.keys()))
        self.rates = np.array([self.yield_dict[m] for m in self.maturities])
        self.discount_map = {
            t: np.exp(-r * t) for t, r in zip(self.maturities, self.rates)
        }

    def discount(self, t):
        """
        Discount factor for maturity t (years)
        """
        if t <= 0:
            return 1.0
        idx = np.searchsorted(self.maturities, t, side='right') - 1
        idx = max(idx, 0)
        tau_k = self.maturities[idx]
        r_k = self.rates[idx]
        df_k = self.discount_map[tau_k]
        tail = t - tau_k
        return df_k * np.exp(-r_k * tail)

    def rate(self, t):
        """
        Effective continuous rate implied by DF(t)
        """
        if t <= 0:
            return 0.0
        df = self.discount(t)
        return -np.log(df) / t

    def plot(self, t_max=30, num_points=300):
        """
        Plot discount curve and implied continuous rate
        """
        T = np.linspace(0.001, t_max, num_points)
        df_vals = [self.discount(t) for t in T]
        r_vals = [self.rate(t) for t in T]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(T, df_vals)
        plt.title("Discount Factor Curve")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("DF(t)")

        plt.subplot(1, 2, 2)
        plt.plot(T, r_vals)
        plt.title("Effective Continuous Rate")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("r_eff(t)")
        plt.tight_layout()
        plt.show()