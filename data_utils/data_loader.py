from BS_utils.yieldCurve import YieldCurve as yc
from BS_utils import maturity_calculator as matur_calc
from BS_utils.implied_dividend import *
from BS_utils.implied_variance_calculator import *
from BS_utils.data_context import DataContext
from datetime import datetime as dt
import polars as pl

def get_quote_date_data(quote_date: dt,

						column_mapping: dict,
						dividend_type: str = 'q_quote', 
						options_data_path: str = './data_utils/CBOE_SPX_Options.csv',
						fred_client: object = None
						) -> pl.DataFrame:
	"""
	Retrieves and processes options data for a specific quote date.

	This function loads options data from a CSV file, filters for the specified quote date,
	and calculates various financial metrics including implied volatilities, forward prices,
	log-moneyness, and time-weighted implied volatilities.

	Parameters
	----------
	quote_date : datetime
		The date for which to retrieve options data.
	column_mapping : dict
		Dictionary mapping logical column names to actual column names in the dataset.
	dividend_type : str, optional
		The type of dividend yield calculation to use ('q_quote' or 'q_iv'), by default 'q_quote'.
	options_data_path : str, optional
		Path to the CSV file containing options data, by default './data_utils/CBOE_SPX_Options.csv'.
	fred_client : object, optional
		Client object for accessing Federal Reserve Economic Data, used for yield curve calculation.

	Returns
	-------
	pandas.DataFrame
		A DataFrame containing filtered options data for the specified quote date with additional
		calculated fields including:
		- mid prices
		- time-to-maturity (T)
		- risk-free rates (r)
		- dividend yields (q_quote, q_iv)
		- forward prices (F)
		- log-moneyness (m)
		- implied volatilities (bid_IV, mid_IV, ask_IV)
		- adjusted implied volatility spreads with intersection and union approaches
		- time-weighted implied volatilities (TIV)

	Raises
	------
	KeyError
		If the 'quote_date' column is not found in the dataset.
	ValueError
		If no data is available for the specified quote date.

	Notes
	-----
	The function applies various filters and calculations to ensure data quality:
	- Filters out expired options (T > 0)
	- Computes implied dividend yields using two different methods
	- Calculates implied volatilities using parallel computation	
	"""
	
	# Read data with Polars
	options_data = pl.read_csv(options_data_path)

	# Extract date only for filtering
	quote_date_str = quote_date.strftime('%Y-%m-%d')
	# Filter data for the quote date
	if 'quote_date' in options_data.columns:
		quote_data = options_data.filter(pl.col('quote_date') == quote_date_str)
	else:
		raise KeyError("Expected 'quote_date' column not found in dataset.")

	if quote_data.is_empty():
		raise ValueError("Calculation stopped: No data available for the specified quote date.")

	# Wrap with DataContext using your logical mapping
	ctx = DataContext(quote_data, column_mapping)
	ctx.set('mid', (ctx.get('ask') + ctx.get('bid')) / 2)

	# Build yield curve for the quote date
	yield_curve = yc(quote_date,fred_client)

	ctx.to_datetime('expiration')
	# Convert 'expiration' to datetime with 9am set as hour
	ctx.update_column("expiration", lambda x: x.replace(hour=9))

	# Compute time-to-maturity using your maturity calculator
	T = matur_calc.calculate_time_to_maturity(quote_date, ctx.get("expiration"), parallel=True)
	ctx.set("T", T)

	# Filter out expired options
	ctx.filter(ctx.col("T") > 0)

	# Get underlying price (assuming it's constant across options)
	S_0 = ctx.get("S_0")[0]

	# Compute risk-free rate r(t) from time to maturity
	r_values = ctx.get("T").map_elements(yield_curve.rate, return_dtype=pl.Float64)
	ctx.set("r", r_values)

	# Compute dividend yield q(t) using implied dividends from quotes
	q_quote = compute_implied_dividends_from_quotes(
		ctx, S_0, 
		min_oi=75, 
		min_ask=0.1, 
		min_bid=0.1)

	ctx.set('q_quote', q_quote)

	# Compute dividend yield q(t) using implied dividends from IV
	q_iv = get_implied_dividend(
		ctx,
		S_0,
		min_oi=75,
		min_ask=0.1,
		min_bid=0.1,
	)

	ctx.set('q_iv', q_iv)
	# Set which dividend type to use
	q_type = dividend_type

	def compute_forward_price(S_0: float, r: pl.Series, q: pl.Series, T: pl.Series) -> pl.Series:
		"""
		Compute forward price F = S_0 * exp((r - q) * T)
		"""
		return S_0 * ((r - q) * T).exp()

	def compute_log_moneyness(K: pl.Series, F: pl.Series) -> pl.Series:
		"""
		Compute log-moneyness log(K / F)
		"""
		return (K / F).log()

	F = compute_forward_price(S_0, ctx.get("r"), ctx.get(q_type), ctx.get("T"))
	ctx.set("F", F)

	log_m = compute_log_moneyness(ctx.get("strike"), F)
	ctx.set("m", log_m)

	log_0 = compute_log_moneyness(ctx.get("S_0"), F)
	ctx.set('m_0',log_0)

	ctx.set('bid_IV', compute_iv_column_parallel(ctx, 'bid', dividend_type=q_type))
	ctx.set('mid_IV', compute_iv_column_parallel(ctx, 'mid', dividend_type=q_type))
	ctx.set('ask_IV', compute_iv_column_parallel(ctx, 'ask', dividend_type=q_type))

	# adj_iv_result = compute_adjusted_iv_spreads(ctx)

	# ctx.set('intersection_type', adj_iv_result['intersection_type'])

	# ctx.set('intersection_bid_IV', adj_iv_result['intersection_bid_IV'])
	# ctx.set('intersection_ask_IV', adj_iv_result['intersection_ask_IV'])
	# ctx.set('intersection_mid_IV', compute_adj_mid_iv_column_parallel(ctx, 'intersection', dividend_type=q_type))

	# ctx.set('union_bid_IV', adj_iv_result['union_bid_IV'])
	# ctx.set('union_ask_IV', adj_iv_result['union_ask_IV'])
	# ctx.set('union_mid_IV', compute_adj_mid_iv_column_parallel(ctx, 'union', dividend_type=q_type))

	ctx.set('bid_TIV', ctx.get('bid_IV')**2 * ctx.get('T'))
	ctx.set('mid_TIV', ctx.get('mid_IV')**2 * ctx.get('T'))
	ctx.set('ask_TIV', ctx.get('ask_IV')**2 * ctx.get('T'))

	# ctx.set('union_bid_TIV', ctx.get('union_bid_IV')**2 * ctx.get('T'))
	# ctx.set('union_mid_TIV', ctx.get('union_mid_IV')**2 * ctx.get('T'))
	# ctx.set('union_ask_TIV', ctx.get('union_ask_IV')**2 * ctx.get('T'))

	# ctx.set('intersection_bid_TIV', ctx.get('intersection_bid_IV')**2 * ctx.get('T'))
	# ctx.set('intersection_mid_TIV', ctx.get('intersection_mid_IV')**2 * ctx.get('T'))
	# ctx.set('intersection_ask_TIV', ctx.get('intersection_ask_IV')**2 * ctx.get('T'))

	return ctx.raw().to_pandas()
