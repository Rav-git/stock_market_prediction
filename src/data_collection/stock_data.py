"""
Module for fetching stock market data from various financial APIs.
"""

import yfinance as yf
import pandas as pd
import requests
import time
import logging
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Class for fetching stock data from Yahoo Finance API."""
    
    def __init__(self, symbols=None):
        """Initialize with stock symbols."""
        self.symbols = symbols or config.STOCK_SYMBOLS
        logger.info(f"Initialized StockDataFetcher with symbols: {self.symbols}")
        
    def get_historical_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch historical data for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period (e.g., '1d', '1mo', '1y')
            interval (str): Data interval (e.g., '1m', '1h', '1d')
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} with period {period} and interval {interval}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            data.reset_index(inplace=True)
            data['Symbol'] = symbol
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_current_price(self, symbol):
        """
        Get the current price of a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Current stock price
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            return data['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
            
    def get_batch_data(self, period="1mo", interval="1d"):
        """
        Fetch data for all symbols in batch.
        
        Returns:
            dict: Dictionary with stock data for each symbol
        """
        result = {}
        for symbol in self.symbols:
            result[symbol] = self.get_historical_data(symbol, period, interval)
            time.sleep(1)  # Avoid rate limiting
        return result

class FinancialNewsFetcher:
    """Class for fetching financial news related to stocks."""
    
    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key or config.NEWS_API_KEY
        
    def get_news_for_symbol(self, symbol, days=7):
        """
        Get news articles related to a stock symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to look back
            
        Returns:
            list: List of news articles
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = ('https://newsapi.org/v2/everything?'
                   f'q={symbol} stock&'
                   f'from={start_date.strftime("%Y-%m-%d")}&'
                   f'to={end_date.strftime("%Y-%m-%d")}&'
                   'sortBy=popularity&'
                   f'apiKey={self.api_key}')
            
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()['articles']
            else:
                logger.error(f"Error fetching news for {symbol}: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Exception while fetching news for {symbol}: {e}")
            return []
            
    def get_batch_news(self, symbols=None, days=7):
        """
        Get news for multiple symbols.
        
        Args:
            symbols (list): List of stock symbols
            days (int): Number of days to look back
            
        Returns:
            dict: Dictionary with news for each symbol
        """
        symbols = symbols or config.STOCK_SYMBOLS
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_news_for_symbol(symbol, days)
            time.sleep(1)  # Avoid rate limiting
        return result

# Example usage
if __name__ == "__main__":
    stock_fetcher = StockDataFetcher()
    data = stock_fetcher.get_historical_data("AAPL", period="1mo")
    print(data.head())
    
    current_price = stock_fetcher.get_current_price("AAPL")
    print(f"Current AAPL price: ${current_price:.2f}")
    
    if config.NEWS_API_KEY != "YOUR_NEWS_API_KEY":
        news_fetcher = FinancialNewsFetcher()
        news = news_fetcher.get_news_for_symbol("AAPL", days=3)
        print(f"Found {len(news)} news articles for AAPL")
    else:
        print("Set your NEWS_API_KEY in config.py to fetch news") 