import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_candlestick(df, ticker):
    if df.empty:
        st.warning("No data available to plot.")
        return

    # Create subplots
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Price Chart", "Daily Returns", "Volatility")
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Add moving averages if they exist
    if 'MA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['MA20'], 
                mode='lines', 
                name='MA20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'MA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['MA50'], 
                mode='lines', 
                name='MA50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

    # Daily returns
    if 'Daily Return' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Daily Return'],
                name='Daily Returns'
            ),
            row=2, col=1
        )

    # Volatility
    if 'Volatility' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Volatility'],
                mode='lines',
                name='Volatility',
                fill='tozeroy'
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        height=800,
        title=f"{ticker} Technical Analysis",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Returns", row=2, col=1)
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def plot_classification_results(y_true, y_pred, y_pred_proba):
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Down', 'Up'],
                   y=['Down', 'Up'],
                   text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='Receiver Operating Characteristic (ROC) Curve'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Prediction Distribution
    st.subheader("Prediction Distribution")
    pred_counts = pd.Series(y_pred).value_counts()
    fig = px.pie(values=pred_counts.values, 
                names=['Down' if x == 0 else 'Up' for x in pred_counts.index],
                title='Prediction Distribution')
    st.plotly_chart(fig, use_container_width=True)

def plot_regression_results(y_true, y_pred):
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.4f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.4f}")
    with col3:
        st.metric("R² Score", f"{r2:.4f}")
    
    # Actual vs Predicted
    st.subheader("Actual vs Predicted Values")
    fig = px.scatter(x=y_true, y=y_pred, 
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title='Actual vs Predicted Values')
    fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], 
                            y=[min(y_true), max(y_true)], 
                            mode='lines', 
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals Plot
    st.subheader("Residuals Analysis")
    residuals = y_true - y_pred
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals Distribution", "Residuals vs Predicted"))
    
    # Residuals distribution
    fig.add_trace(go.Histogram(x=residuals, name='Residuals', nbinsx=30), row=1, col=1)
    
    # Residuals vs Predicted
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], 
                            mode='lines', name='Zero Residual',
                            line=dict(color='red', dash='dash')), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Residual Value", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Value", row=1, col=2)
    fig.update_yaxes(title_text="Residual Value", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Error Over Time
    st.subheader("Prediction Error Over Time")
    error_over_time = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': residuals
    }, index=y_true.index if hasattr(y_true, 'index') else range(len(y_true)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=error_over_time.index, y=error_over_time['Error'], 
                            mode='lines', name='Prediction Error'))
    fig.add_trace(go.Scatter(x=error_over_time.index, y=[0]*len(error_over_time), 
                            mode='lines', name='Zero Error',
                            line=dict(color='red', dash='dash')))
    fig.update_layout(xaxis_title='Time', yaxis_title='Prediction Error')
    st.plotly_chart(fig, use_container_width=True)

def plot_bootstrap_samples(bootstrap_samples, data_index, window_size):
    """Visualize bootstrap sampling for the first two trees"""
    if len(bootstrap_samples) < 1:
        return

    # allow 1 or 2 trees supplied
    n_to_plot = min(2, len(bootstrap_samples))

    fig = make_subplots(
        rows=n_to_plot, cols=1,
        subplot_titles=[f"Tree {i+1}: Bootstrap Sampling" for i in range(n_to_plot)],
        vertical_spacing=0.12
    )

    for i in range(n_to_plot):
        # Unpack expected tuple: (X_boot, y_boot, window_start, bootstrap_indices)
        try:
            X_boot, y_boot, window_start, bootstrap_indices = bootstrap_samples[i]
        except ValueError:
            # backward compatibility: if bootstrap_samples had old shape (X_boot,y_boot,start)
            X_boot, y_boot, window_start = bootstrap_samples[i]
            # fallback: attempt to reconstruct bootstrap_indices as a range within window
            bootstrap_indices = np.arange(window_start, min(window_start + len(X_boot), len(data_index)))

        n_total = len(data_index)
        # Create a binary array indicating which samples (absolute indices) were selected
        selected = np.zeros(n_total, dtype=int)
        # bootstrap_indices should be absolute indices; guard bounds
        for idx in np.unique(bootstrap_indices):
            if 0 <= int(idx) < n_total:
                selected[int(idx)] = 1

        # Add trace for selected samples
        fig.add_trace(
            go.Scatter(
                x=data_index,
                y=selected,
                mode='markers',
                name=f'Tree {i+1} Selected Samples',
                marker=dict(size=6)
            ),
            row=i+1, col=1
        )

        # Add window region: use absolute indices window_start..window_end
        window_end_idx = min(window_start + window_size - 1, n_total - 1)
        x0 = data_index[window_start]
        x1 = data_index[window_end_idx]

        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="lightgreen", opacity=0.25, line_width=0,
            annotation_text=f"Window: {window_start}-{window_end_idx}",
            annotation_position="top left",
            row=i+1, col=1
        )

        # Update y-axis for each subplot
        fig.update_yaxes(
            title_text="Selected (1) / Not Selected (0)",
            range=[-0.1, 1.1],
            tickvals=[0, 1],
            row=i+1, col=1
        )

        # correct xref/yref for subplot annotation: first subplot uses 'x domain' / 'y domain'
        if i == 0:
            xref_str = "x domain"
            yref_str = "y domain"
        else:
            xref_str = f"x{i+1} domain"
            yref_str = f"y{i+1} domain"

        # Add text annotation with stats (positioned within the subplot domain)
        n_selected = int(selected.sum())
        selection_rate = n_selected / n_total if n_total > 0 else 0.0

        fig.add_annotation(
            x=0.02, y=0.98,
            xref=xref_str, yref=yref_str,
            text=f"Selected: {n_selected}/{n_total} ({selection_rate:.1%})",
            showarrow=False,
            bgcolor="white",
            row=i+1, col=1
        )

    fig.update_layout(
        height=300 * n_to_plot,
        title_text="Bootstrap Sampling Visualization",
        showlegend=False
    )

    fig.update_xaxes(title_text="Date", row=n_to_plot, col=1)

    st.plotly_chart(fig, use_container_width=True)