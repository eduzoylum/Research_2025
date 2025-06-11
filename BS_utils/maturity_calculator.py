import pandas_market_calendars as mcal
import pandas as pd
import polars as pl
from datetime import datetime as dt
from joblib import Parallel, delayed

def calculate_time_to_maturity(quote_date: dt, maturity_dates, alpha=0.7, parallel=False):
    """
    Calculate time to maturity for one or more maturity dates.

    Parameters:
        quote_date (datetime): The quote date.
        maturity_dates (datetime or Series): A single maturity date or Series of them.
        alpha (float): Weighting factor for trading vs non-trading hours.
        parallel (bool): Whether to run in parallel for vector input.

    Returns:
        float or Series: Time to maturity in days.
    """
    nyse = mcal.get_calendar('CBOE_Index_Options')

    if not isinstance(maturity_dates, pd.Series):  
        if isinstance(maturity_dates, pl.Series):
            maturity_dates = maturity_dates.to_pandas()
        elif isinstance(maturity_dates, list):
            maturity_dates = pd.Series(maturity_dates)
        elif isinstance(maturity_dates, dt):
            maturity_dates = pd.Series([maturity_dates])
        else:
            raise ValueError("maturity_dates must be a datetime, list, or Pandas Series.")        

    end_date = maturity_dates.max() if isinstance(maturity_dates, pd.Series) else maturity_dates
    schedule = nyse.schedule(start_date=quote_date, end_date=end_date).copy()
    schedule['market_open'] = schedule['market_open'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    schedule['market_close'] = schedule['market_close'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

    if isinstance(maturity_dates, pd.Series):
        if parallel:
            results = Parallel(n_jobs=-1)(
                delayed(_calculate_single_time_to_maturity)(quote_date, m, schedule, alpha)
                for m in maturity_dates
            )
            return pd.Series(results, index=maturity_dates.index)
        else:
            return maturity_dates.apply(
                lambda m: _calculate_single_time_to_maturity(quote_date, m, schedule, alpha)
            )
    else:
        return _calculate_single_time_to_maturity(quote_date, maturity_dates, schedule, alpha)


def _expired_calculate_time_to_maturity(quote_date: dt, maturity_dates, alpha=0.7):
    """
    Calculate time to maturity for one or more maturity dates.

    Parameters:
        quote_date (datetime): The quote date.
        maturity_dates (datetime or Series): A single maturity date or a Pandas Series of maturity dates.
        alpha (float): Weighting factor for trading vs non-trading hours.

    Returns:
        float or Series: Time to maturity (T) in days for each maturity date.
    """
    # Create a calendar for the CBOE
    nyse = mcal.get_calendar('CBOE_Index_Options')

    # Check if maturity_dates is a Series, if not convert it
    if not isinstance(maturity_dates, pd.Series):  
        if isinstance(maturity_dates, pl.Series):
            maturity_dates = maturity_dates.to_pandas()
        elif isinstance(maturity_dates, list):
            maturity_dates = pd.Series(maturity_dates)
        elif isinstance(maturity_dates, dt):
            maturity_dates = pd.Series([maturity_dates])
        else:
            raise ValueError("maturity_dates must be a datetime, list, or Pandas Series.")

    # Define a time range
    start_date = quote_date
    end_date = maturity_dates.max() if isinstance(maturity_dates, pd.Series) else maturity_dates

    # Get the schedule within the specified time range
    schedule = nyse.schedule(start_date=start_date, end_date=end_date).copy()

    # Convert the schedule to EST time zone
    schedule['market_open'] = schedule['market_open'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    schedule['market_close'] = schedule['market_close'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

    # Handle vectorized or single maturity date calculation
    if isinstance(maturity_dates, pd.Series):
        results = maturity_dates.apply(
            lambda maturity: _calculate_single_time_to_maturity(quote_date, maturity, schedule, alpha)
        )
        return results
    else:
        return _calculate_single_time_to_maturity(quote_date, maturity_dates, schedule, alpha)


def calculate_trading_and_non_trading_hours(current_time, maturity, schedule):
    """
    Calculate trading and non-trading hours between two time points.

    Parameters:
        current_time (datetime): The starting time.
        maturity (datetime): The ending time.
        schedule (DataFrame): Market schedule with 'market_open' and 'market_close' columns.
    Returns:
        tuple: Trading hours and non-trading hours as floats.
    """
    # Filter the schedule for the relevant dates
    relevant_schedule = schedule[(schedule['market_open'] >= current_time) & (schedule['market_close'] <= maturity)]

    # Vectorized calculation of trading hours
    start_times = relevant_schedule['market_open'].clip(lower=current_time)
    end_times = relevant_schedule['market_close'].clip(upper=maturity)
    trading_hours = ((end_times - start_times).dt.total_seconds() / 3600).sum()

    # Calculate total hours between the two time points
    total_hours = (maturity - current_time).total_seconds() / 3600

    # Non-trading hours
    non_trading_hours = total_hours - trading_hours

    return trading_hours, non_trading_hours


def _calculate_single_time_to_maturity(quote_date, maturity_date, schedule, alpha):
    """
    Helper function to calculate time to maturity for a single maturity date.

    Parameters:
        quote_date (datetime): The quote date.
        maturity_date (datetime): The maturity date.
        schedule (DataFrame): Market schedule.
        alpha (float): Weighting factor for trading vs non-trading hours.

    Returns:
        float: Time to maturity (T) in days.
    """
    trading_hours, non_trading_hours = calculate_trading_and_non_trading_hours(quote_date, maturity_date, schedule)
    T = trading_hours * alpha / 1638 + non_trading_hours * (1 - alpha) / 7122
    return T
