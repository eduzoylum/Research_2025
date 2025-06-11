import numpy as np
import polars as pl
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

def get_implied_dividend(ctx, S_0: float, **kwargs) -> pl.Series:
    """
    Estimate implied dividend yield q across all maturities using BS model fitting.

    Parameters
    ----------
    ctx : DataContext
        Context-wrapped Polars DataFrame.
    S_0 : float
        Spot price of the underlying.
    kwargs : dict
        Passed to `get_implied_q_for_maturity` (e.g., iv_key, min_oi).

    Returns
    -------
    pl.Series
        Series of estimated q values aligned with ctx rows.
    """
    T_values = ctx.get("T").unique().to_list()
    q_per_maturity = {}

    for T_val in T_values:
        maturity_ctx = ctx.filter(ctx.col("T") == T_val)
        q_hat = get_implied_q_for_maturity(maturity_ctx, S_0, **kwargs)
        q_per_maturity[T_val] = q_hat

    T_series = ctx.get("T")
    q_series = T_series.map_elements(lambda t: q_per_maturity.get(t, None), return_dtype=pl.Float64)
    return q_series

def get_implied_q_for_maturity(ctx, S_0: float, 
							min_oi: int = 75,
							min_bid : float = 0.1,
							min_ask : float = 0.1)-> float:
	"""
	Estimate implied dividend yield q for a single maturity using mid prices and BS model.

	Parameters
	----------
	ctx : DataContext
		Context-wrapped Polars DataFrame containing a single maturity.
	S_0 : float
		Spot price of the underlying.
	min_oi : int
		Minimum open interest to include an option in the calibration.
	min_bid : float
		Minimum bid price to include an option in the calibration.
	min_ask : float
		Minimum ask price to include an option in the calibration.
	
	Returns
	-------
	float
		Estimated implied dividend yield q.
	"""
	ctx = ctx.filter(
		condition=(
			(ctx.col("open_interest") > min_oi) &
			(ctx.col("bid") > min_bid) &
			(ctx.col("ask") > min_ask)
			)
	)
	if ctx.is_empty():
		return np.nan

	# Extract values as NumPy arrays
	K = ctx.get("strike").to_numpy()
	r = ctx.get("r").to_numpy()
	T = ctx.get("T").to_numpy()  # Single maturity
	iv = ctx.get('iv_source').to_numpy()
	bid = ctx.get("bid").to_numpy()
	ask = ctx.get("ask").to_numpy()
	OI = ctx.get("open_interest").to_numpy()

	opt_type = ctx.get("option_type")

	mid_prices = (bid + ask) / 2
	weights = OI / OI.sum()

	def objective(q_array):
		q = q_array[0]  # Extract scalar from array
		modeled = np.array([
			bf.BS_Call(S_0, K[i], r[i], T[i], iv[i], q)
			if opt_type[i] == 'C' else
			bf.BS_Put(S_0, K[i], r[i], T[i], iv[i], q)
			for i in range(len(K))
		])
		return np.mean(weights * (modeled - mid_prices) ** 2)

	return fmin(objective, [0.01], xtol=1e-4,  maxiter=1e6, maxfun=1e6, disp=False)[0]

def compute_implied_dividends_from_quotes(ctx, S_0: float, 
                                        min_oi: int = 50, 
                                        min_bid: float = 0.1,
                                        min_ask: float = 0.1) -> pl.Series:
    """
    Compute implied dividend yields using call-put parity and Polars + DataContext.

    Parameters
    ----------
    ctx : DataContext
        Context-wrapped Polars DataFrame with standardized column access.
    S_0 : float
        Underlying spot price.
    min_oi : int
        Minimum open interest for filtering options.
    Returns
    -------
    pl.Series
        Vector of implied dividend yields per row.
    """

    maturities = ctx.get("T").unique().to_list()
    q_vals = []

    for maturity in maturities:

        subset = ctx.filter(
            condition=(
                (ctx.col("T") == maturity) & # Filter by maturity
                (ctx.col('open_interest') > min_oi) & # Minimum open interest
                (ctx.col('bid') > min_bid) & # Minimum bid price
                (ctx.col('ask') > min_ask) # Minimum ask price
            )
        )

        if subset.is_empty():
            q_vals.append((maturity, None))
            continue

        calls = subset.filter(
            ctx.col("option_type") == "C"
        )
        puts  = subset.filter(
            ctx.col("option_type") == "P"
        )

        merged = calls.df.join(
            puts.df,
            on=[
                ctx.col("strike"),
                ctx.col("T"),
                ctx.col("r")
            ],
            suffix="_put"
        )

        if merged.is_empty():
            q_vals.append((maturity, np.nan))
            continue


        K,T,r = merged.select(
            ctx.col("strike"),
            ctx.col("T"),
            ctx.col("r"),
        )

        C_ask, C_bid, OI_call = merged.select(
            ctx.col("ask"),
            ctx.col("bid"),
            ctx.col("open_interest")
        )

        P_ask, P_bid, OI_put = merged.select(
            pl.col(ctx.col_map['ask']+"_put"),
            pl.col(ctx.col_map['bid']+"_put"),
            pl.col(ctx.col_map['open_interest']+"_put")
        )

        inside1 = ((C_ask - P_bid + K * (-r * T).exp()) / S_0).clip(lower_bound=1e-10)
        inside2 = ((C_bid - P_ask + K * (-r * T).exp()) / S_0).clip(lower_bound=1e-10)

        q1 = -(inside1.log()) / T
        q2 = -(inside2.log()) / T

        w_q = (q1 * OI_put + q2 * OI_call) / (OI_call + OI_put)
        weights = OI_call + OI_put
        q_t = (w_q * weights).sum() / weights.sum()

        q_vals.append((maturity, q_t))

    #Map result back to original df length
    t_col = ctx.get("T")
    q_out = t_col.map_elements(lambda x: dict(q_vals).get(x, None),return_dtype=pl.Float64)
    #ctx.set("q_imp", q_out)
    return q_out

class DividendCurve_smooth:
	def __init__(self, T_q_dcx, q_ind_label='q_quote'):
		"""
		Parameters
		----------
		T_q_df : DataContext for Polars DataFrame
			Must contain columns: 'T' (maturities) and 'q_imp' (dividends)
		q_imp_ind : str
			Logical column name for dividend yields (default: 'q_imp')
		"""		
		# Collapse multiple rows with same T by averaging q_imp

		grouped_terms = ctx.group_by('T').agg(
								ctx.col('q_quote').mean()
							).sort('T')

		self.T = grouped_terms[:,0].to_numpy()
		self.q = grouped_terms[:,1].to_numpy()
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
    def __init__(self, T_q_dcx, q_ind_label='q_quote'):
        """
        Parameters
        ----------
        T_q_df : DataContext for Polars DataFrame
            Must contain columns: 'T' (maturities) and 'q_imp' (dividends)
        q_imp_ind : str
            Logical column name for dividend yields (default: 'q_imp')
        """		
        self.T = T_q_dcx.get("T").to_numpy()
        self.q = T_q_dcx.get(q_ind_label).to_numpy()
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
        """Effective dividend yield implied by A(t)"""
        if t <= 0:
            return 0.0
        A = self.appreciation(t)
        return -np.log(A) / t

    def plot(self, t_max=6.0, num_points=300):
        """Plot A(t) and implied q(t) over a time grid"""
        T_grid = np.linspace(0, t_max, num_points)
        A_vals = np.array([self.appreciation(t) for t in T_grid])
        q_vals = np.array([self.q_at(t) for t in T_grid])
        A_breaks = np.array([self.appreciation(t) for t in self.T])

        plt.figure(figsize=(12, 6))

        # Plot A(t)
        plt.subplot(1, 2, 1)
        plt.plot(T_grid, A_vals, label="A(t)", color='black')
        plt.scatter(self.T, A_breaks, color='red', label="Breakpoints", zorder=5, s=3)
        plt.title("Stock Appreciation Factor")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("A(t)")
        plt.grid(True)
        plt.legend()

        # Plot q(t)
        plt.subplot(1, 2, 2)
        plt.plot(T_grid, q_vals, label="Implied q(t)", color='black')
        plt.scatter(self.T, self.q, color='red', zorder=5, s=3)
        plt.title("Dividend Yield Curve")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("q(t)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
