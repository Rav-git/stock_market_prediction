"""
Simplified Streamlit web application for stock market analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from itertools import cycle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from data_collection.stock_data import StockDataFetcher

# Set page configuration
st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize components
@st.cache_resource
def get_stock_fetcher():
    return StockDataFetcher()

# Cache for data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, period):
    stock_fetcher = get_stock_fetcher()
    return stock_fetcher.get_historical_data(symbol, period=period)

# Function to calculate moving averages and RSI
def calculate_technical_indicators(stock_data):
    df = stock_data.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Function to make simple predictions
def make_simple_predictions(stock_data, days_ahead):
    """
    Make more sophisticated predictions using statistical methods.
    
    Args:
        stock_data (DataFrame): Historical stock data
        days_ahead (int): Number of days to predict ahead
        
    Returns:
        DataFrame: Predictions with confidence intervals
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    import pmdarima as pm
    
    # Get last date and price
    last_date = stock_data['Date'].iloc[-1]
    last_price = stock_data['Close'].iloc[-1]
    
    try:
        # Use auto-ARIMA for better predictions
        prices = stock_data['Close'].values
        
        # Check if the series is stationary
        result = adfuller(prices)
        is_stationary = result[1] < 0.05
        
        # Use auto ARIMA to find best parameters
        model = pm.auto_arima(
            prices,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=None,
            trace=False
        )
        
        # Forecast future prices
        forecast, conf_int = model.predict(n_periods=days_ahead, return_conf_int=True, alpha=0.05)
        
        # Create future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create predictions dataframe
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': forecast,
            'Lower_Bound': conf_int[:, 0],
            'Upper_Bound': conf_int[:, 1]
        })
        
        return predictions
        
    except Exception as e:
        st.warning(f"Advanced prediction model failed: {str(e)}. Using fallback method.")
        
        # Fallback to a more robust method
        # Calculate average daily return for the last 30 days with exponential weighting
        returns = stock_data['Close'].pct_change().dropna()
        
        # Use exponentially weighted mean to give more importance to recent returns
        exp_returns = returns.ewm(span=10).mean().tail(30)
        avg_return = exp_returns.mean()
        
        # Add some persistence - recent trend has more impact
        recent_trend = returns.tail(5).mean()
        avg_return = 0.7 * avg_return + 0.3 * recent_trend
        
        # Predict future prices with compound growth
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        predicted_prices = [last_price * (1 + avg_return) ** (i+1) for i in range(days_ahead)]
        
        # Calculate confidence bounds based on volatility
        # Use EWMA for volatility estimate (more responsive to recent volatility)
        volatility = returns.ewm(span=20).std().tail(30).mean()
        confidence_factor = 1.96  # 95% confidence interval
        
        # Increasing confidence interval for further predictions
        upper_bounds = [price * (1 + volatility * confidence_factor * (1 + i*0.1)) for i, price in enumerate(predicted_prices)]
        lower_bounds = [price * (1 - volatility * confidence_factor * (1 + i*0.1)) for i, price in enumerate(predicted_prices)]
        
        # Create predictions dataframe
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predicted_prices,
            'Upper_Bound': upper_bounds,
            'Lower_Bound': lower_bounds
        })
        
        return predictions

# App header
st.title("Stock Market Analysis")
st.markdown("""
This application analyzes stock market trends using historical price data from Yahoo Finance.
""")

# Sidebar for inputs
st.sidebar.header("Settings")

# Stock symbol selection
symbol_options = config.STOCK_SYMBOLS
custom_symbol = st.sidebar.text_input("Enter Custom Stock Symbol:", "")
if custom_symbol:
    selected_symbol = custom_symbol.upper()
else:
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol:", symbol_options)

# Time period selection
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
selected_period_label = st.sidebar.selectbox("Select Time Period:", list(period_options.keys()))
selected_period = period_options[selected_period_label]

# Days to predict ahead
days_ahead = st.sidebar.slider("Days to Predict Ahead:", 1, 30, 5)

# Add multi-stock comparison section
st.sidebar.header("Stock Comparison")
compare_stocks = st.sidebar.checkbox("Enable Stock Comparison")

comparison_symbols = []
if compare_stocks:
    comparison_input = st.sidebar.text_input("Enter stock symbols separated by commas (e.g., MSFT,GOOG,AMZN)")
    if comparison_input:
        # Parse and validate symbols
        comparison_symbols = [sym.strip().upper() for sym in comparison_input.split(",")]
        # Add the main selected symbol if not already in the list
        if selected_symbol not in comparison_symbols:
            comparison_symbols = [selected_symbol] + comparison_symbols
        
        st.sidebar.caption(f"Comparing: {', '.join(comparison_symbols)}")

# Create tabs for different views outside the button click handler
tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Indicators", "Prediction", "Stock Comparison"])

# Initialize or declare variables used later
stock_data = None
predictions = None
stock_data_with_indicators = None
comparison_data = {}

# Display initial content in tabs before any data is loaded
with tab1:
    st.info("Click 'Analyze Stock' to view price chart")
    
with tab2:
    st.info("Click 'Analyze Stock' to view technical indicators")
    
with tab3:
    st.info("Click 'Analyze Stock' to view predictions")
    
with tab4:
    st.info("Enable stock comparison in the sidebar, add stock symbols, and click 'Analyze Stock' to compare")

# Load data when user clicks the button
if st.sidebar.button("Analyze Stock"):
    with st.spinner(f"Fetching data for {selected_symbol}..."):
        # Fetch data
        stock_data = fetch_stock_data(selected_symbol, selected_period)
        
        # Check if stock data is valid
        if stock_data.empty:
            st.error(f"Could not fetch data for symbol {selected_symbol}. Please check if the symbol is valid.")
            st.stop()
            
        # Calculate technical indicators
        stock_data_with_indicators = calculate_technical_indicators(stock_data)
        
        # Make predictions
        predictions = make_simple_predictions(stock_data, days_ahead)
        
        # Fetch comparison data if needed
        comparison_data = {}
        if compare_stocks and comparison_symbols:
            for symbol in comparison_symbols:
                with st.spinner(f"Fetching data for {symbol}..."):
                    comp_data = fetch_stock_data(symbol, selected_period)
                    if not comp_data.empty:
                        comparison_data[symbol] = comp_data
                    else:
                        st.warning(f"Could not fetch data for {symbol}")
        
        # Display stock info
        st.header(f"Stock Analysis: {selected_symbol}")
        
        # Current price and stats
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2]
        price_change = ((current_price / prev_price) - 1) * 100
        price_color = "green" if price_change >= 0 else "red"
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
        with col2:
            st.metric("Opening Price", f"${stock_data['Open'].iloc[-1]:.2f}")
        with col3:
            st.metric("High", f"${stock_data['High'].iloc[-1]:.2f}")
        with col4:
            st.metric("Low", f"${stock_data['Low'].iloc[-1]:.2f}")
        
        # Create tabs for different views
        # Note: We're keeping this line but the tabs are now defined outside the if statement
        # tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Indicators", "Prediction", "Stock Comparison"])
        
        with tab1:
            st.subheader("Stock Price History")
            
            # Create interactive chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ))
            
            # Add volume as a bar chart at the bottom
            fig.add_trace(go.Bar(
                x=stock_data['Date'],
                y=stock_data['Volume'],
                name='Volume',
                marker_color='rgba(200, 200, 200, 0.5)',
                yaxis='y2'
            ))
            
            # Customize layout
            fig.update_layout(
                title=f"Price History for {selected_symbol}",
                xaxis_title='Date',
                yaxis_title='Price',
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(
                        visible=True
                    ),
                    type='date'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Technical Indicators")
            
            # Create figure for technical indicators
            fig = go.Figure()
            
            # Add price
            fig.add_trace(go.Scatter(
                x=stock_data_with_indicators['Date'],
                y=stock_data_with_indicators['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=stock_data_with_indicators['Date'],
                y=stock_data_with_indicators['MA5'],
                mode='lines',
                name='5-day MA',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=stock_data_with_indicators['Date'],
                y=stock_data_with_indicators['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='orange')
            ))
            
            fig.add_trace(go.Scatter(
                x=stock_data_with_indicators['Date'],
                y=stock_data_with_indicators['MA50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='green')
            ))
            
            # Customize layout
            fig.update_layout(
                title=f"Moving Averages for {selected_symbol}",
                xaxis_title='Date',
                yaxis_title='Price',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI Chart
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=stock_data_with_indicators['Date'],
                y=stock_data_with_indicators['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # Add overbought/oversold lines
            fig_rsi.add_shape(
                type="line",
                x0=stock_data_with_indicators['Date'].iloc[0],
                y0=70,
                x1=stock_data_with_indicators['Date'].iloc[-1],
                y1=70,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_rsi.add_shape(
                type="line",
                x0=stock_data_with_indicators['Date'].iloc[0],
                y0=30,
                x1=stock_data_with_indicators['Date'].iloc[-1],
                y1=30,
                line=dict(color="green", width=2, dash="dash")
            )
            
            # Customize layout
            fig_rsi.update_layout(
                title=f"Relative Strength Index (RSI) for {selected_symbol}",
                xaxis_title='Date',
                yaxis_title='RSI',
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        with tab3:
            st.subheader("Price Prediction")
            
            # Create figure for predictions
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add predictions
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Close'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, symbol='circle', opacity=0.8)
            ))
            
            # Add confidence bounds
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Upper_Bound'],
                mode='lines',
                name='95% Confidence Upper Bound',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Lower_Bound'],
                mode='lines',
                name='95% Confidence Lower Bound',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            ))
            
            # Add vertical line at the last historical date
            fig.add_shape(
                type="line",
                x0=stock_data['Date'].iloc[-1],
                y0=min(stock_data['Close'].min(), predictions['Lower_Bound'].min()) * 0.95,
                x1=stock_data['Date'].iloc[-1],
                y1=max(stock_data['Close'].max(), predictions['Upper_Bound'].max()) * 1.05,
                line=dict(color="black", width=2, dash="dash")
            )
            
            # Add annotation
            fig.add_annotation(
                x=stock_data['Date'].iloc[-1],
                y=stock_data['Close'].max(),
                text="Prediction Start",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )
            
            # Customize layout
            fig.update_layout(
                title=f"Price Prediction for {selected_symbol}",
                xaxis_title='Date',
                yaxis_title='Price ($)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            st.subheader("Prediction Summary")
            
            last_price = stock_data['Close'].iloc[-1]
            predicted_price = predictions['Predicted_Close'].iloc[-1]
            percent_change = ((predicted_price / last_price) - 1) * 100
            
            direction = "Increase" if percent_change > 0 else "Decrease"
            direction_color = "green" if percent_change > 0 else "red"
            
            # Show prediction details in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${last_price:.2f}", 
                    delta=f"{price_change:.2f}% (1 day)"
                )
                
            with col2:
                st.metric(
                    f"Predicted ({predictions['Date'].iloc[-1].strftime('%b %d')})", 
                    f"${predicted_price:.2f}", 
                    delta=f"{percent_change:.2f}%",
                    delta_color="normal"
                )
                
            with col3:
                mid_idx = len(predictions) // 2
                mid_price = predictions['Predicted_Close'].iloc[mid_idx]
                mid_change = ((mid_price / last_price) - 1) * 100
                mid_date = predictions['Date'].iloc[mid_idx].strftime('%b %d')
                
                st.metric(
                    f"Mid-term ({mid_date})",
                    f"${mid_price:.2f}",
                    delta=f"{mid_change:.2f}%",
                    delta_color="normal"
                )
            
            # Add prediction insights based on technical indicators
            st.subheader("Prediction Insights")
            
            # RSI analysis
            current_rsi = stock_data_with_indicators['RSI'].iloc[-1]
            rsi_status = ""
            if current_rsi > 70:
                rsi_status = "Overbought - potential reversal downward"
            elif current_rsi < 30:
                rsi_status = "Oversold - potential reversal upward"
            else:
                rsi_status = "Neutral"
                
            # Moving average analysis
            ma5 = stock_data_with_indicators['MA5'].iloc[-1]
            ma20 = stock_data_with_indicators['MA20'].iloc[-1]
            ma50 = stock_data_with_indicators['MA50'].iloc[-1]
            
            ma_signals = []
            if ma5 > ma20:
                ma_signals.append("Short-term bullish (MA5 > MA20)")
            else:
                ma_signals.append("Short-term bearish (MA5 < MA20)")
                
            if ma20 > ma50:
                ma_signals.append("Mid-term bullish (MA20 > MA50)")
            else:
                ma_signals.append("Mid-term bearish (MA20 < MA50)")
                
            # Volatility analysis
            recent_volatility = stock_data['Close'].pct_change().tail(20).std() * 100
            
            with st.expander("Technical Indicators Analysis"):
                indicators_col1, indicators_col2 = st.columns(2)
                
                with indicators_col1:
                    st.markdown(f"**RSI (14):** {current_rsi:.2f}")
                    st.markdown(f"**Status:** {rsi_status}")
                    st.markdown(f"**Recent Volatility:** {recent_volatility:.2f}%")
                    
                with indicators_col2:
                    st.markdown(f"**Moving Averages:** ")
                    for signal in ma_signals:
                        st.markdown(f"- {signal}")
            
            # Add prediction table with more details
            st.subheader("Detailed Predictions")
            
            # Add day of week to predictions
            predictions['Day'] = predictions['Date'].dt.day_name()
            predictions['Date'] = predictions['Date'].dt.strftime('%Y-%m-%d')
            
            # Calculate daily changes
            predictions['Daily Change (%)'] = [0] + [((predictions['Predicted_Close'].iloc[i] / predictions['Predicted_Close'].iloc[i-1]) - 1) * 100 for i in range(1, len(predictions))]
            
            # Round values for display
            display_df = predictions.copy()
            display_df['Predicted_Close'] = display_df['Predicted_Close'].round(2)
            display_df['Upper_Bound'] = display_df['Upper_Bound'].round(2)
            display_df['Lower_Bound'] = display_df['Lower_Bound'].round(2)
            display_df['Daily Change (%)'] = display_df['Daily Change (%)'].round(2)
            
            # Reorder columns for better presentation
            display_df = display_df[['Date', 'Day', 'Predicted_Close', 'Daily Change (%)', 'Lower_Bound', 'Upper_Bound']]
            
            # Display the table
            st.dataframe(
                display_df,
                column_config={
                    "Predicted_Close": st.column_config.NumberColumn(
                        "Predicted Price ($)",
                        format="$%.2f",
                    ),
                    "Lower_Bound": st.column_config.NumberColumn(
                        "Lower Bound ($)",
                        format="$%.2f",
                    ),
                    "Upper_Bound": st.column_config.NumberColumn(
                        "Upper Bound ($)",
                        format="$%.2f",
                    ),
                    "Daily Change (%)": st.column_config.NumberColumn(
                        "Daily Change (%)",
                        format="%.2f%%",
                    ),
                },
                use_container_width=True
            )

        # The tab4 block should be indented to be part of the if statement
        with tab4:
            if not compare_stocks or not comparison_symbols or len(comparison_symbols) <= 1:
                st.info("Enable stock comparison in the sidebar and add stock symbols to compare")
            elif not comparison_data or len(comparison_data) <= 1:
                st.info("Couldn't fetch comparison data for the selected symbols. Please try again or select different symbols.")
            else:
                st.subheader("Stock Price Comparison")
                
                # Normalize prices option
                normalize = st.checkbox("Normalize prices to compare percentage changes", value=True)
                
                # Create comparison figure
                fig = go.Figure()
                
                # Add each stock to the plot
                for symbol in comparison_symbols:
                    if symbol in comparison_data:
                        df = comparison_data[symbol]
                        
                        # Normalize if selected
                        if normalize:
                            y_values = (df['Close'] / df['Close'].iloc[0]) * 100
                            y_axis_title = "Normalized Price (%)"
                        else:
                            y_values = df['Close']
                            y_axis_title = "Price ($)"
                            
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=y_values,
                            mode='lines',
                            name=symbol,
                            line=dict(width=2)
                        ))
                
                # Customize layout
                fig.update_layout(
                    title="Stock Price Comparison",
                    xaxis_title='Date',
                    yaxis_title=y_axis_title,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison metrics
                st.subheader("Performance Metrics")
                
                # Calculate performance metrics
                metrics_df = pd.DataFrame()
                
                for symbol in comparison_symbols:
                    if symbol in comparison_data:
                        df = comparison_data[symbol]
                        
                        # Calculate returns for different periods
                        last_price = df['Close'].iloc[-1]
                        first_price = df['Close'].iloc[0]
                        
                        # Return calculation
                        total_return = ((last_price / first_price) - 1) * 100
                        
                        # Calculate volatility (standard deviation of daily returns)
                        daily_returns = df['Close'].pct_change().dropna()
                        volatility = daily_returns.std() * 100
                        
                        # Calculate average daily volume
                        avg_volume = df['Volume'].mean()
                        
                        # Add to dataframe
                        metrics_df = pd.concat([metrics_df, pd.DataFrame({
                            'Symbol': [symbol],
                            'Current Price': [last_price],
                            f'Return ({selected_period_label})': [total_return],
                            'Daily Volatility (%)': [volatility],
                            'Avg Daily Volume': [avg_volume]
                        })])
                
                # Format and display the metrics table
                metrics_df = metrics_df.reset_index(drop=True)
                
                st.dataframe(
                    metrics_df,
                    column_config={
                        'Current Price': st.column_config.NumberColumn(
                            'Current Price ($)',
                            format="$%.2f"
                        ),
                        f'Return ({selected_period_label})': st.column_config.NumberColumn(
                            f'Return ({selected_period_label})',
                            format="%.2f%%"
                        ),
                        'Daily Volatility (%)': st.column_config.NumberColumn(
                            'Daily Volatility (%)',
                            format="%.2f%%"
                        ),
                        'Avg Daily Volume': st.column_config.NumberColumn(
                            'Avg Daily Volume',
                            format="%d"
                        )
                    },
                    use_container_width=True
                )
                
                # Volume comparison
                st.subheader("Volume Comparison")
                
                # Create volume comparison figure
                vol_fig = go.Figure()
                
                # Add each stock's volume to the plot
                for symbol in comparison_symbols:
                    if symbol in comparison_data:
                        df = comparison_data[symbol]
                        
                        vol_fig.add_trace(go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            name=f"{symbol} Volume",
                            opacity=0.7
                        ))
                
                # Customize volume layout
                vol_fig.update_layout(
                    title="Trading Volume Comparison",
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(vol_fig, use_container_width=True)
                
                # Correlation matrix
                st.subheader("Price Correlation Matrix")
                
                # Create a dataframe with all closing prices
                corr_df = pd.DataFrame()
                
                for symbol in comparison_symbols:
                    if symbol in comparison_data:
                        df = comparison_data[symbol]
                        corr_df[symbol] = df['Close']
                
                # Calculate correlation
                corr_matrix = corr_df.corr()
                
                # Plot correlation matrix
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    aspect="equal"
                )
                fig.update_layout(title="Price Correlation Matrix")
                
                st.plotly_chart(fig, use_container_width=True)

# This should be the last code in the file
if __name__ == "__main__":
    pass  # Streamlit runs the script directly, no need for a main() function 