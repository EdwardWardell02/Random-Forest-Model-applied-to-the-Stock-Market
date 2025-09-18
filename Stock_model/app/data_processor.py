import pandas as pd
import numpy as np

def process_stock_data(df):
    if df.empty:
        return df

    df = df.sort_index().copy()

    # Calculate daily returns
    df['Daily Return'] = df['Close'].pct_change()

    # Always create MA20 and MA50 columns, even if incomplete
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    # Volatility (annualised)
    df['Volatility'] = df['Daily Return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
    df['Price_Drop_Indicator'] = (df['Close'] < df['Open']).astype(int)
    df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
    df['Below_MA20'] = (df['Close'] < df['MA20']).astype(int)
    df['Below_MA50'] = (df['Close'] < df['MA50']).astype(int)

    return df

