from BS_utils import bs_functions as bf
from joblib import Parallel, delayed
from scipy.optimize import brentq
import polars as pl
import numpy as np

def compute_iv_column_parallel(ctx, price_type: str, n_jobs: int = -1, dividend_type: str = 'q_quote') -> pl.Series:
	"""
	Compute implied volatilities in parallel for the given price type ('bid', 'ask', 'mid').

	Parameters
	----------
	ctx : DataContext 
		DataContext object for the given quote date
	price_type : str
		One of 'bid', 'ask', 'mid'
	n_jobs : int
		Number of parallel jobs to run (-1 means all cores)

	Returns
	-------
	pl.Series
		Implied volatility values
	"""
	df = ctx.raw()
	K = ctx.get("strike").to_numpy()
	T = ctx.get("T").to_numpy()
	r = ctx.get("r").to_numpy()
	q = ctx.get(dividend_type).to_numpy()
	S = ctx.get("S_0").to_numpy()
	price = ctx.get(price_type).to_numpy()

	opt_type = ctx.get("option_type")

	def compute_single_iv(i):
		try:
			if opt_type[i] == "C":
				return brentq(
					lambda sigma: bf.BS_Call(S[i], K[i], r[i], T[i], sigma, q[i]) - price[i],
					a=1e-4, b=5.0, xtol=1e-6, maxiter=1000, disp=False
				)
			else:
				return brentq(
					lambda sigma: bf.BS_Put(S[i], K[i], r[i], T[i], sigma, q[i]) - price[i],
					a=1e-4, b=5.0, xtol=1e-6, maxiter=1000, disp=False
				)
		except (ValueError, RuntimeError):
			return 1e-4

	ivs = Parallel(n_jobs=n_jobs)(
		delayed(compute_single_iv)(i) for i in range(len(df))
	)

	return pl.Series(f"{price_type}_IV", ivs)

def compute_adjusted_iv_spreads(ctx):
	"""
	Compute adjusted implied volatility (IV) spreads by combining call and put option data.

	This function calculates several versions of implied volatility bid/ask spreads 
	by combining information from corresponding call and put options:

	1. Intersection spread: Uses the higher bid IV and lower ask IV between 
		call and put options to potentially create tighter spreads.
	2. Union spread: Uses the lower bid IV and higher ask IV between call and put 
		options to create wider but more conservative spreads.
	3. Fallback values: When the intersection produces invalid spreads (bid > ask), 
		falls back to either call (when K > F) or put (when K <= F) values.

	Parameters
	----------
	ctx : Context
		A context object containing:
		- raw: Method that returns the raw dataframe
		- col_map: Dictionary mapping column names for 'strike', 'T' (time to maturity), 
			'r' (risk-free rate), 'option_type', 'bid_IV', 'ask_IV', and 'F' (forward price)

	Returns
	-------
	DataFrame
		Original dataframe augmented with new columns:
		- bid_IV_intersection, ask_IV_intersection: Bid/ask IV from intersection method
		- bid_IV_union, ask_IV_union: Bid/ask IV from union method
		- intersection_type: 'normal' if intersection is valid, 'fallback' if using fallback values
	"""
	df = ctx.raw()
	strike_col = ctx.col_map["strike"]
	T_col = ctx.col_map["T"]
	r_col = ctx.col_map["r"]
	opt_col = ctx.col_map["option_type"]
	iv_bid_col = ctx.col_map["bid_IV"]
	iv_ask_col = ctx.col_map["ask_IV"]
	forward_col_actual = ctx.col_map["F"]

	# Separate calls and puts
	calls = df.filter(pl.col(opt_col) == "C")
	puts = df.filter(pl.col(opt_col) == "P")

	# Join on strike, T, r
	joined = calls.join(puts, on=[strike_col, T_col, r_col], suffix="_put")

	# Now extract relevant columns
	joined = joined.with_columns([
		pl.col(iv_bid_col).alias("C_bid"),
		pl.col(iv_ask_col).alias("C_ask"),
		pl.col(iv_bid_col + "_put").alias("P_bid"),
		pl.col(iv_ask_col + "_put").alias("P_ask"),
		pl.col(strike_col).alias("strike"),
		pl.col(forward_col_actual).alias("F")
	])

	# Intersection and union
	joined = joined.with_columns([
		pl.max_horizontal("C_bid", "P_bid").alias("iv_bid_inter"),
		pl.min_horizontal("C_ask", "P_ask").alias("iv_ask_inter"),
		pl.min_horizontal("C_bid", "P_bid").alias("union_bid_IV"),
		pl.max_horizontal("C_ask", "P_ask").alias("union_ask_IV")
	])

	# Step 1: Add fallback_call first so we can use it below
	joined = joined.with_columns(
		(pl.col("strike") > pl.col("F")).alias("fallback_call")
	)

	# Step 2: Now use fallback_call in dependent expressions
	joined = joined.with_columns([
		pl.when(pl.col("fallback_call")).then(pl.col("C_bid")).otherwise(pl.col("P_bid")).alias("fb_bid"),
		pl.when(pl.col("fallback_call")).then(pl.col("C_ask")).otherwise(pl.col("P_ask")).alias("fb_ask")
	])

	# Step 3: Add valid_intersection in its own step
	joined = joined.with_columns(
		(pl.col("iv_bid_inter") <= pl.col("iv_ask_inter")).alias("valid_intersection")
	)

	# Step 4: Now use it in the final logic
	joined = joined.with_columns([
		pl.when(pl.col("valid_intersection")).then(pl.col("iv_bid_inter")).otherwise(pl.col("fb_bid")).alias("intersection_bid_IV"),
		pl.when(pl.col("valid_intersection")).then(pl.col("iv_ask_inter")).otherwise(pl.col("fb_ask")).alias("intersection_ask_IV"),
		pl.when(pl.col("valid_intersection")).then(pl.lit("normal")).otherwise(pl.lit("fallback")).alias("intersection_type")
	])

	# Prepare call and put outputs
	base_cols = [strike_col, T_col]
	extra_cols = [
		"intersection_bid_IV",
		"intersection_ask_IV",
		"union_bid_IV",
		"union_ask_IV",
		"intersection_type"
	]
	call_rows = joined.select(
		[pl.col(c) for c in base_cols + extra_cols]
	)

	return ctx.raw().join(
		call_rows, on=['strike', 'T'], how='left'
	)


def compute_adj_mid_iv_column_parallel(ctx, join_type: str, n_jobs: int = -1, dividend_type: str = 'q_quote') -> pl.Series:
	"""
	Compute adjusted mid-price implied volatilities in parallel for the given adjustment type ('intersection', 'union').

	Parameters
	----------
	ctx : DataContext 
		DataContext object for the given quote date
	join_type : str
		One of 'intersection', 'union'
	n_jobs : int
		Number of parallel jobs to run (-1 means all cores)

	Returns
	-------
	pl.Series
		Implied volatility values
	"""
	df = ctx.raw()
	K = ctx.get("strike").to_numpy()
	T = ctx.get("T").to_numpy()
	r = ctx.get("r").to_numpy()
	q = ctx.get(dividend_type).to_numpy()
	S = ctx.get("S_0").to_numpy()

	opt_type = ctx.get("option_type")

	adj_up_lim = ctx.get(f"{join_type}_ask_IV").to_numpy()
	adj_low_lim = ctx.get(f"{join_type}_bid_IV").to_numpy()
	adj_low_price = [
		bf.BS_Call(S[i], K[i], r[i], T[i], adj_low_lim[i], q[i]) if opt_type[i] == "C" else
		bf.BS_Put(S[i], K[i], r[i], T[i], adj_low_lim[i], q[i])
		for i in range(len(df))
	]
	adj_up_price = [
		bf.BS_Call(S[i], K[i], r[i], T[i], adj_up_lim[i], q[i]) if opt_type[i] == "C" else
		bf.BS_Put(S[i], K[i], r[i], T[i], adj_up_lim[i], q[i])
		for i in range(len(df))
	]
	price = (np.array(adj_low_price) + np.array(adj_up_price)) / 2


	def compute_single_iv(i):
		try:
			if opt_type[i] == "C":
				return brentq(
					lambda sigma: bf.BS_Call(S[i], K[i], r[i], T[i], sigma, q[i]) - price[i],
					a=1e-4, b=5.0, xtol=1e-6, maxiter=1000, disp=False
				)
			else:
				return brentq(
					lambda sigma: bf.BS_Put(S[i], K[i], r[i], T[i], sigma, q[i]) - price[i],
					a=1e-4, b=5.0, xtol=1e-6, maxiter=1000, disp=False
				)
		except (ValueError, RuntimeError):
			return 1e-4

	ivs = Parallel(n_jobs=n_jobs)(
		delayed(compute_single_iv)(i) for i in range(len(df))
	)

	return pl.Series(f"{join_type}_mid_IV", ivs)

