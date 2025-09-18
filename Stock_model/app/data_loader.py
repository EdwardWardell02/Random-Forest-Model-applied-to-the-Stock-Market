import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import os
import time
import random
import streamlit as st

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

ALPHA_VANTAGE_API_KEY = "D8NZVFIM53WG653X"

def get_stock_filename(ticker):
    return os.path.join(DATA_DIR, f"{ticker}.csv")

def load_cached_data(ticker):
    """Load all cached data for a ticker from CSV"""
    filename = get_stock_filename(ticker)
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            df.index.name = "Date"
            return df.sort_index()
        except Exception as e:
            st.warning(f"Error reading cache file for {ticker}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_data_to_cache(ticker, df):
    filename = get_stock_filename(ticker)
    df.sort_index().to_csv(filename)

def process_alpha_vantage_data(data):
    df = data[0]

    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
    )

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].replace({",": ""}, regex=True)
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def update_or_download_data(ticker, start_date, end_date):
    """
    Return cached data if it covers the requested range.
    Only request Alpha Vantage if CSV is missing data.
    """
    cached = load_cached_data(ticker)

    # STEP 1: Check if cached data covers the requested range
    if not cached.empty:
        cached_start = cached.index.min().date()
        cached_end = cached.index.max().date()
        end_date = end_date.date()
        start_date = start_date.date()
        if cached_start <= start_date <= end_date <= cached_end:
            # Cache fully covers requested range
            return cached

    # STEP 2: If we get here, CSV is missing data → request from Alpha Vantage
    #st.info(f"Fetching data from Alpha Vantage for {ticker} (cache missing dates)...")
    try:
        time.sleep(random.uniform(1.5, 3.0))  # jitter for rate limits
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        data, meta = ts.get_daily(symbol=ticker, outputsize="full")
        new_data = process_alpha_vantage_data((data, meta))

        # Combine with cached data if any
        if not cached.empty:
            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_data

        save_data_to_cache(ticker, combined)
        
        # Load the updated CSV file to ensure we have the latest data
        return load_cached_data(ticker)

    except ValueError as e:
        # If API call fails (rate limit), fall back to whatever CSV we have
        #st.warning(f"Alpha Vantage API request failed: {e}")
        if not cached.empty:
            #st.info("Returning cached data only (may not cover full range).")
            return cached
        else:
            return pd.DataFrame()

# Remove the streamlit cache decorator to ensure we always get fresh data
def fetch_stock_data(ticker, start_date, end_date):
    df = update_or_download_data(ticker, start_date, end_date)
    if df.empty:
        return df
    
    # Ensure we're only returning data within the requested range
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask]