"""
Streamlit web application for stock market prediction and sentiment analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import config and modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import modules
from data_collection.stock_data import StockDataFetcher, FinancialNewsFetcher
from data_collection.social_data import TwitterDataFetcher
from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from prediction.stock_predictor import StockPredictor
from visualization.visualizer import StockVisualizer

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction with Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize components
@st.cache_resource
def get_stock_fetcher():
    return StockDataFetcher()

@st.cache_resource
def get_news_fetcher():
    return FinancialNewsFetcher()

@st.cache_resource
def get_twitter_fetcher():
    return TwitterDataFetcher()

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentAnalyzer(method='vader')

@st.cache_resource
def get_stock_predictor():
    return StockPredictor(model_type='rf')

@st.cache_resource
def get_visualizer():
    return StockVisualizer()

# Cache for data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, period):
    stock_fetcher = get_stock_fetcher()
    return stock_fetcher.get_historical_data(symbol, period=period)

@st.cache_data(ttl=3600)
def fetch_news_data(symbol, days):
    news_fetcher = get_news_fetcher()
    return news_fetcher.get_news_for_symbol(symbol, days=days)

@st.cache_data(ttl=3600)
def fetch_twitter_data(symbol, max_results):
    twitter_fetcher = get_twitter_fetcher()
    return twitter_fetcher.get_stock_tweets(symbol, max_results=max_results)

# Function to analyze sentiment
def analyze_sentiment(news_data, social_data):
    sentiment_analyzer = get_sentiment_analyzer()
    
    # Analyze news
    news_sentiment = sentiment_analyzer.analyze_news(news_data)
    
    # Analyze social media
    social_sentiment = sentiment_analyzer.analyze_tweets(social_data)
    
    # Get overall sentiment scores by date
    sentiment_by_date = None
    
    if not news_sentiment.empty:
        # Group by date and calculate mean sentiment
        news_sentiment['date'] = pd.to_datetime(news_sentiment['publishedAt'])
        news_grouped = news_sentiment.groupby(news_sentiment['date'].dt.date).agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        }).reset_index()
        
        sentiment_by_date = news_grouped
    
    if not social_sentiment.empty:
        # Group by date
        social_sentiment['date'] = pd.to_datetime(social_sentiment['created_at'])
        social_grouped = social_sentiment.groupby(social_sentiment['date'].dt.date).agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        }).reset_index()
        
        # Combine with news sentiment or use as is
        if sentiment_by_date is not None:
            # Merge news and social sentiment
            sentiment_by_date = pd.merge(
                sentiment_by_date,
                social_grouped,
                on='date',
                how='outer',
                suffixes=('_news', '_social')
            )
            
            # Calculate weighted average (giving more weight to news)
            sentiment_by_date['compound'] = sentiment_by_date['compound_news'] * 0.6 + sentiment_by_date['compound_social'] * 0.4
            sentiment_by_date['positive'] = sentiment_by_date['positive_news'] * 0.6 + sentiment_by_date['positive_social'] * 0.4
            sentiment_by_date['negative'] = sentiment_by_date['negative_news'] * 0.6 + sentiment_by_date['negative_social'] * 0.4
            sentiment_by_date['neutral'] = sentiment_by_date['neutral_news'] * 0.6 + sentiment_by_date['neutral_social'] * 0.4
            
            # Keep only the combined columns
            sentiment_by_date = sentiment_by_date[['date', 'compound', 'positive', 'negative', 'neutral']]
        else:
            sentiment_by_date = social_grouped
    
    return news_sentiment, social_sentiment, sentiment_by_date

# Function to make predictions
@st.cache_data(ttl=3600)
def make_predictions(stock_data, sentiment_data, days_ahead):
    predictor = get_stock_predictor()
    
    # Train model (simple model for demo)
    predictor.train(stock_data, sentiment_data)
    
    # Make predictions
    return predictor.predict(stock_data, sentiment_data, days_ahead)

# App header
st.title("Stock Market Prediction with Sentiment Analysis")
st.markdown("""
This application analyzes stock market trends using historical price data 
and sentiment analysis from financial news and social media.
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

# News days to fetch
news_days = st.sidebar.slider("Days of News to Analyze:", 7, 60, 30)

# Social media posts to fetch
social_posts = st.sidebar.slider("Social Media Posts to Analyze:", 10, 200, 50)

# Include sentiment in prediction
include_sentiment = st.sidebar.checkbox("Include Sentiment in Prediction", value=True)

# Load data when user clicks the button
if st.sidebar.button("Analyze Stock"):
    with st.spinner(f"Fetching data for {selected_symbol}..."):
        # Fetch data
        stock_data = fetch_stock_data(selected_symbol, selected_period)
        
        # Check if stock data is valid
        if stock_data.empty:
            st.error(f"Could not fetch data for symbol {selected_symbol}. Please check if the symbol is valid.")
            st.stop()
            
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
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Sentiment Analysis", "Prediction", "Raw Data"])
        
        with tab1:
            st.subheader("Stock Price History")
            
            # Create interactive chart
            visualizer = get_visualizer()
            fig = visualizer.plot_interactive_stock_prices(stock_data, title=f"Price History for {selected_symbol}")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Sentiment Analysis")
            
            # Fetch sentiment data
            with st.spinner("Analyzing sentiment from news and social media..."):
                news_data = fetch_news_data(selected_symbol, news_days)
                twitter_data = fetch_twitter_data(selected_symbol, social_posts)
                
                news_sentiment, social_sentiment, sentiment_by_date = analyze_sentiment(news_data, twitter_data)
                
                if sentiment_by_date is not None and not sentiment_by_date.empty:
                    # Display sentiment chart
                    fig = visualizer.plot_interactive_sentiment(
                        sentiment_by_date, 
                        stock_data, 
                        title=f"Sentiment Analysis for {selected_symbol}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display overall sentiment
                    latest_sentiment = sentiment_by_date.iloc[-1]
                    overall_sentiment = "Positive" if latest_sentiment['compound'] > 0.05 else "Negative" if latest_sentiment['compound'] < -0.05 else "Neutral"
                    sentiment_color = "green" if overall_sentiment == "Positive" else "red" if overall_sentiment == "Negative" else "gray"
                    
                    st.markdown(f"### Overall Sentiment: <span style='color:{sentiment_color}'>{overall_sentiment}</span>", unsafe_allow_html=True)
                    
                    # Display sentiment components
                    sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
                    with sent_col1:
                        st.metric("Compound Score", f"{latest_sentiment['compound']:.2f}")
                    with sent_col2:
                        st.metric("Positive", f"{latest_sentiment['positive']:.2f}")
                    with sent_col3:
                        st.metric("Negative", f"{latest_sentiment['negative']:.2f}")
                    with sent_col4:
                        st.metric("Neutral", f"{latest_sentiment['neutral']:.2f}")
                    
                    # Display recent news
                    if not news_sentiment.empty:
                        st.subheader("Recent News Articles")
                        for i, article in news_sentiment.sort_values('publishedAt', ascending=False).head(5).iterrows():
                            sentiment = "positive" if article['sentiment'] == 'positive' else "negative" if article['sentiment'] == 'negative' else "neutral"
                            color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
                            st.markdown(f"**{article['title']}** - <span style='color:{color}'>{sentiment.capitalize()} ({article['compound']:.2f})</span>", unsafe_allow_html=True)
                            st.markdown(f"*{article['description']}*")
                            st.markdown(f"Source: [{article['source']['name']}]({article['url']}) | {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d')}")
                            st.markdown("---")
                    
                    # Display recent tweets
                    if not social_sentiment.empty:
                        st.subheader("Recent Social Media Posts")
                        for i, tweet in social_sentiment.sort_values('created_at', ascending=False).head(5).iterrows():
                            sentiment = "positive" if tweet['sentiment'] == 'positive' else "negative" if tweet['sentiment'] == 'negative' else "neutral"
                            color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
                            st.markdown(f"**{tweet['text']}**")
                            st.markdown(f"<span style='color:{color}'>{sentiment.capitalize()} ({tweet['compound']:.2f})</span> | {pd.to_datetime(tweet['created_at']).strftime('%Y-%m-%d')}", unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.warning("No sentiment data available for analysis.")
                    
        with tab3:
            st.subheader("Stock Price Prediction")
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                sentiment_data_for_prediction = sentiment_by_date if include_sentiment else None
                predictions = make_predictions(stock_data, sentiment_data_for_prediction, days_ahead)
                
                # Show predictions
                fig = visualizer.plot_interactive_predictions(
                    stock_data, 
                    predictions, 
                    title=f"Price Prediction for {selected_symbol}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction metrics
                current_price = stock_data['Close'].iloc[-1]
                future_price = predictions['Predicted_Close'].iloc[-1]
                price_change = ((future_price / current_price) - 1) * 100
                direction = "up" if price_change > 0 else "down"
                direction_color = "green" if direction == "up" else "red"
                
                st.markdown(f"### Prediction: Price will go <span style='color:{direction_color}'>{direction}</span> by {abs(price_change):.2f}%", unsafe_allow_html=True)
                
                # Display prediction table
                st.subheader("Predicted Prices")
                predictions['Date'] = predictions['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(predictions)
                
        with tab4:
            st.subheader("Raw Data")
            
            # Display stock data table
            st.dataframe(stock_data)
            
            # Add download button
            csv = stock_data.to_csv(index=False)
            st.download_button(
                label="Download stock data as CSV",
                data=csv,
                file_name=f"{selected_symbol}_stock_data.csv",
                mime="text/csv"
            )

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application is a demonstration of stock market prediction using 
machine learning and sentiment analysis from financial news and social media.

**Note:** The predictions are for educational purposes only and should not be 
used for actual investment decisions.
""")

# Check if API keys are set
st.sidebar.markdown("---")
st.sidebar.markdown("### API Status")

if config.NEWS_API_KEY != "YOUR_NEWS_API_KEY":
    st.sidebar.success("✓ News API configured")
else:
    st.sidebar.error("✗ News API not configured")
    
if config.TWITTER_API_KEY != "YOUR_TWITTER_API_KEY":
    st.sidebar.success("✓ Twitter API configured")
else:
    st.sidebar.error("✗ Twitter API not configured")
    
if config.FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
    st.sidebar.success("✓ Finnhub API configured") 
else:
    st.sidebar.error("✗ Finnhub API not configured") 