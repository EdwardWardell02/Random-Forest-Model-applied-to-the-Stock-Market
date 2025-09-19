import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor as SklearnRFR
from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from sklearn.metrics import classification_report, confusion_matrix

from app.data_loader import fetch_stock_data
from app.data_processor import process_stock_data
from app.update_visualiser import plot_candlestick, plot_classification_results, plot_regression_results, plot_bootstrap_samples
from app.Random_Forest_Model_Classifier import RandomForestClassifier
from app.Random_Forest_Model_Regressor import RandomForestRegressor

# Set page configuration with a professional theme
st.set_page_config(
    page_title="Stock Analysis Dashboard", 
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1F2730;
        color: #FAFAFA;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 24px;
        font-weight: bold;
        color: #00D4AA;
        margin-bottom: 20px;
    }
    
    /* Sidebar subheader */
    .sidebar-subheader {
        font-size: 18px;
        font-weight: bold;
        color: #00D4AA;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #00D4AA;
        color: #0E1117;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        background-color: #00B894;
        color: #0E1117;
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background-color: #00D4AA;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>div {
        background-color: #2C3E50;
        color: #FAFAFA;
    }
    
    /* Radio button styling */
    .stRadio>div {
        background-color: #2C3E50;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #1F2730;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #00D4AA;
    }
    
    /* Tabs styling */
    .stTabs>div>div>div {
        background-color: #1F2730;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1F2730;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Header section
col1, col2 = st.columns([1, 3])
with col1:
    # Add Stock image for a professional look.
    st.image("https://cdn.mos.cms.futurecdn.net/JKqxBpmH3e95tdXXLrNyPZ.jpg", width=300)
with col2:
    st.title("Stock Analysis Dashboard")
    st.markdown("**Advanced predictive analytics for stock market forecasting**")

# Initialise tabs for stock and forex.
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'SP500'

# Sidebar with enhanced design, also allow sidebar to fold in and out.
with st.sidebar:
    st.markdown('<p class="sidebar-header">Market Selection</p>', unsafe_allow_html=True)
    market_type = st.radio("Select Market:", ["SP500", "Forex"], index=0)

    # Make a safe logic stament for the market type, this helps later 
    st.session_state.selected_tab = market_type

    # Given the market type a select box with the relevant tickers is made. 
    if market_type == "SP500":
        st.markdown('<p class="sidebar-header">Stock Selection</p>', unsafe_allow_html=True)
        stock_tickers = {
                "Apple": "AAPL",
                "Microsoft": "MSFT",
                "Google": "GOOGL",
                "Amazon": "AMZN",
                "Tesla": "TSLA",
                "Meta": "META",
                "NVIDIA": "NVDA",
                "Netflix": "NFLX",
                "Intel": "INTC",
                "IBM": "IBM"
        } 
        selected_ticker = st.selectbox(
            "Select a stock:",
            list(stock_tickers.keys()),
            index=0
        )
        ticker_symbol = stock_tickers[selected_ticker]

    else:
        st.markdown('<p class="sidebar-header">Forex Selection</p>', unsafe_allow_html=True)
        forex_tickers = {
                "EUR/USD": "EURUSD",
                "GBP/USD": "GBPUSD",
                "USD/JPY": "USDJPY",
                "AUD/USD": "AUDUSD",
                "USD/CAD": "USDCAD",
                "USD/CHF": "USDCHF",
                "NZD/USD": "NZDUSD",
                "EUR/GBP": "EURGBP",
                "EUR/JPY": "EURJPY",
                "GBP/JPY": "GBPJPY"
            }

        selected_ticker = st.selectbox(
            "Select a stock:",
            list(forex_tickers.keys()),
            index=0
            )

        ticker_symbol = forex_tickers[selected_ticker]
    

    st.markdown('<p class="sidebar-subheader">Time Span</p>', unsafe_allow_html=True)
    
    # Time span selection
    time_span = st.selectbox(
        "Select time span:",
        [
            "1 Day",
            "5 Days",
            "1 Week",
            "2 Weeks",
            "1 Month",
            "3 Months",
            "6 Months",
            "1 Year",
            "2 Years",
            "5 Years",
            "10 Years"
        ],
        index=3,
    )

    st.markdown('<p class="sidebar-subheader">Model Type</p>', unsafe_allow_html=True)
    
    model_type = st.radio(
        "",
        ["Classification", "Regression"],
        index=0,
        help="Classifier predicts price direction (up/down), Regressor predicts actual price values."
    )

    # Date calculation (all of the data is from the current date and then back the chosen time span.
    end_date = datetime.today()
    if time_span == "1 Day":
        start_date = end_date - timedelta(days=1)
    elif time_span == "5 Days":
        start_date = end_date - timedelta(days=5)
    elif time_span == "1 Week":
        start_date = end_date - timedelta(days=7)
    elif time_span == "2 Weeks":
        start_date = end_date - timedelta(days=14)
    elif time_span == "1 Month":
        start_date = end_date - timedelta(days=30)
    elif time_span == "3 Months":
        start_date = end_date - timedelta(days=90)
    elif time_span == "6 Months":
        start_date = end_date - timedelta(days=180)
    elif time_span == "1 Year":
        start_date = end_date - timedelta(days=365)
    elif time_span == "2 Years":
        start_date = end_date - timedelta(days=730)
    elif time_span == "5 Years":
        start_date = end_date - timedelta(days=1825)
    elif time_span == "10 Years":
        start_date = end_date - timedelta(days=3650)  # Approximate 10 years
    else:
        start_date = end_date - timedelta(days=365)

    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    st.markdown(f"**Date Range:** {start_date.date()} to {end_date.date()}")

# Fetch and process stock data for the chosen ticker.
raw_data = fetch_stock_data(ticker_symbol, start_date, end_date)

# Display data and analysis
if raw_data.empty:
    st.warning("No data available for the selected stock and time span or Alpha Vantage API request limit reached.")
else:
    # Drop any rows with missing values
    raw_data = raw_data.dropna(axis=0, how='any')
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Market Data", "ML Model", "Performance"])
    
    with tab1:
        # Display latest price metrics
        latest_data = raw_data.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Open", f"${latest_data['Open']:.2f}")
        with col2:
            st.metric("High", f"${latest_data['High']:.2f}")
        with col3:
            st.metric("Low", f"${latest_data['Low']:.2f}")
        with col4:
            st.metric("Close", f"${latest_data['Close']:.2f}")
        
        st.metric("Volume", f"{latest_data['Volume']:,.0f}")
        
        # Process and display stock data
        processed_data = process_stock_data(raw_data)
        
        if processed_data.empty:
            st.warning("Not enough data to compute technical indicators.")
        else:
            plot_candlestick(processed_data, ticker_symbol)
    
    with tab2:
        st.header("Machine Learning Model")
        
        n_samples = len(processed_data)
        
        # Adjust window (for bootstrapping window size) size slider based on available data
        max_window = min(100000, n_samples)
        min_window = min(10, n_samples)
        
        available_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'Daily Return', 'MA20', 'MA50', 'Volatility', 
                             'Price_Drop_Indicator', 'Volume_Spike', 'Below_MA20', 'Below_MA50']
        
        selected_features = st.multiselect(
            "Select features for the model:",
            options=available_features,
            default=['Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        # Create target variable, different for classifier and regressor (classifier looks to see  the movement is up or down for close and regressor tried to determine the top price of the stock for that day). 
        if model_type == "Classification":
            processed_data['Target'] = (processed_data['Close'].shift(-1) > processed_data['Close']).astype(int)
        else:
            processed_data['Target'] = processed_data['High'].shift(-1)

        processed_data = processed_data.dropna(subset=['Target'])

        # Ensure features don't contain NaN
        if len(selected_features) > 0:
            processed_data = processed_data.dropna(subset=selected_features)
        
        X = processed_data[selected_features]
        y = processed_data['Target']
        
        # Show feature analysis if multiple features selected
        if len(selected_features) > 1:
            with st.expander("Feature Analysis", expanded=False, width=1200):
                st.subheader("Feature Correlation Matrix")
                corr = X[selected_features].corr()
                
                fig = px.imshow(
                    corr,
                    zmin=-1, zmax=1,
                    color_continuous_scale='RdBu_r',
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if len(selected_features) > 6:
                    st.markdown("##### Feature Pair Analysis")
                    
                    # Create a grid of plots with pagination
                    features_per_page = 6
                    n_pages = (len(selected_features) + features_per_page - 1) // features_per_page
                    
                    page = st.selectbox("Select page", options=range(1, n_pages+1), format_func=lambda x: f"Page {x}")
                    
                    start_idx = (page - 1) * features_per_page
                    end_idx = min(start_idx + features_per_page, len(selected_features))
                    page_features = selected_features[start_idx:end_idx]
                    
                    # Create a grid of scatter plots
                    n_cols = 2
                    n_rows = (len(page_features) + n_cols - 1) // n_cols
                    
                    for i in range(0, len(page_features), n_cols):
                        cols = st.columns(n_cols)
                        for j, feature in enumerate(page_features[i:i+n_cols]):
                            if j < len(cols):
                                with cols[j]:
                                    # Show distribution of this feature
                                    fig = px.histogram(processed_data, x=feature, title=f"Distribution of {feature}")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show this feature against all others
                                    for other_feature in page_features:
                                        if other_feature != feature:
                                            small_fig = px.scatter(
                                                processed_data, 
                                                x=feature, 
                                                y=other_feature,
                                                title=f"{feature} vs {other_feature}",
                                                height=300
                                            )
                                            small_fig.update_traces(marker=dict(size=3, opacity=0.6))
                                            st.plotly_chart(small_fig, use_container_width=True)
                else:
                    # For fewer features, show the full matrix
                    st.markdown("##### Scatter Matrix")
                    fig = px.scatter_matrix(
                        X[selected_features],
                        dimensions=selected_features,
                        height=800,
                        title="Scatter Matrix"
                    )
                    fig.update_traces(
                        marker=dict(size=3, opacity=0.6, line=dict(width=0.2, color='DarkSlateGrey'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Hyperparameter tuning section
        st.subheader("Model Configuration")
        # Use grid search to suggest optimal hyperparameters

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Trees", min_value=0, max_value=2000, value=1000, step=100)
            max_depth = st.slider("Max Depth", min_value=1, max_value=100, value=5, step=1)
            min_samples_split = st.slider("Min Samples to Split", min_value=2, max_value=20, value=5, step=1)
        
        with col2:
            min_samples_leaf = st.slider("Min Samples per Leaf", min_value=1, max_value=20, value=2, step=1)
            max_features = st.slider("Max Features", min_value=1, max_value=len(selected_features), 
                                    value=min(5, len(selected_features)), step=1)
            max_samples = st.slider("Max Samples", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
            window_size = st.slider("Window Size", min_value=min_window, max_value=max_window, 
                                    value=min(30, n_samples), step=5)
            
            if model_type == "Classification":
                criterion = st.radio("Split Criterion", ["gini", "entropy"], index=0)
        
        # Train button
        if st.button("Train Model", use_container_width=True):
            if len(selected_features) == 0:
                st.error("Please select at least one feature for the model.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Training Progress: {progress*100:.1f}%")
                
                if model_type == "Classification":
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        max_samples=max_samples,
                        random_state=42,
                        window_size=window_size,
                        criterion=criterion
                    )
                    
                    rf.fit(X, y, progress_callback=update_progress)
                    y_pred = rf.predict(X)
                    y_pred_proba = rf.predict_proba(X)
                    accuracy = np.mean(y_pred == y)
                    
                    status_text.text(f"Training completed! Accuracy: {accuracy*100:.2f}%")
                    st.success(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")
                    
                    # Compare with sklearn
                    sklearn_rfc = SklearnRFC(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        max_samples=max_samples,
                        random_state=42,
                    )
                    sklearn_rfc.fit(X, y)
                    sklearn_y_pred = sklearn_rfc.predict(X)
                    sklearn_accuracy = np.mean(sklearn_y_pred == y)
                    
                    st.info(f"Scikit-learn Comparison: {sklearn_accuracy*100:.2f}% accuracy")
                    
                    # Show bootstrap samples
                    st.subheader("Bootstrap Sampling")
                    plot_bootstrap_samples(rf.bootstrap_samples_[:2], X.index, window_size)
                    
                    # Plot results
                    plot_classification_results(y, y_pred, y_pred_proba)
                    
                else:
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        max_samples=max_samples,
                        random_state=42,
                        window_size=window_size
                    )
                    
                    rf.fit(X, y, progress_callback=update_progress)
                    y_pred = rf.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    
                    status_text.text(f"Training completed! MSE: {mse:.4f}")
                    st.success(f"Model trained successfully! MSE: {mse:.4f}, R²: {r2:.4f}")
                    
                    # Compare with sklearn
                    sklearn_rfr = SklearnRFR(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        max_samples=max_samples,
                        random_state=42,
                    )
                    sklearn_rfr.fit(X, y)
                    sklearn_y_pred = sklearn_rfr.predict(X)
                    sklearn_mse = mean_squared_error(y, sklearn_y_pred)
                    sklearn_r2 = r2_score(y, sklearn_y_pred)
                    
                    st.info(f"Scikit-learn Comparison: MSE: {sklearn_mse:.4f}, R²: {sklearn_r2:.4f}")
                    
                    # Show bootstrap samples
                    st.subheader("Bootstrap Sampling")
                    plot_bootstrap_samples(rf.bootstrap_samples_[:2], X.index, window_size)
                    
                    # Plot results
                    plot_regression_results(y, y_pred)
    
    with tab3:
        st.header("Model Performance")
        
        if 'rf' in locals():
            if model_type == "Classification":
                st.subheader("Classification Report")
                
                st.write("**Custom Model Performance:**")
                st.text(classification_report(y, y_pred))
                
                st.write("**Scikit-learn Model Performance:**")
                st.text(classification_report(y, sklearn_y_pred))
                
                # Confusion matrix
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                cm_custom = confusion_matrix(y, y_pred)
                sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title('Custom Model Confusion Matrix')
                
                cm_sklearn = confusion_matrix(y, sklearn_y_pred)
                sns.heatmap(cm_sklearn, annot=True, fmt='d', cmap='Blues', ax=ax2)
                ax2.set_title('Scikit-learn Model Confusion Matrix')
                
                st.pyplot(fig)
                
            else:
                st.subheader("Regression Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Custom Model:**")
                    st.metric("MSE", f"{mse:.4f}")
                    st.metric("MAE", f"{mae:.4f}")
                    st.metric("R² Score", f"{r2:.4f}")
                
                with col2:
                    sklearn_mae = mean_absolute_error(y, sklearn_y_pred)
                    sklearn_r2 = r2_score(y, sklearn_y_pred)
                    
                    st.write("**Scikit-learn Model:**")
                    st.metric("MSE", f"{sklearn_mse:.4f}")
                    st.metric("MAE", f"{sklearn_mae:.4f}")
                    st.metric("R² Score", f"{sklearn_r2:.4f}")
                
                # Residual plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                residuals_custom = y - y_pred
                ax1.scatter(y_pred, residuals_custom, alpha=0.5)
                ax1.axhline(y=0, color='r', linestyle='--')
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Residuals')
                ax1.set_title('Custom Model Residual Plot')
                
                residuals_sklearn = y - sklearn_y_pred
                ax2.scatter(sklearn_y_pred, residuals_sklearn, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Scikit-learn Model Residual Plot')
                
                st.pyplot(fig)
        else:
            st.info("Train a model first to see performance metrics")
