"""
Module for fetching social media data related to stocks.
"""

import tweepy
import pandas as pd
import logging
import sys
import os
import time
from datetime import datetime, timedelta

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwitterDataFetcher:
    """Class for fetching tweets related to stocks."""
    
    def __init__(self):
        """Initialize Twitter API with credentials from config."""
        self.api_key = config.TWITTER_API_KEY
        self.api_secret = config.TWITTER_API_SECRET
        self.access_token = config.TWITTER_ACCESS_TOKEN
        self.access_secret = config.TWITTER_ACCESS_SECRET
        self.client = self._get_client()
        
    def _get_client(self):
        """Create and return a Twitter API client."""
        if (self.api_key == "YOUR_TWITTER_API_KEY" or 
            self.api_secret == "YOUR_TWITTER_API_SECRET"):
            logger.warning("Twitter API credentials not set. Twitter data fetching is disabled.")
            return None
            
        try:
            client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret
            )
            return client
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
            return None
            
    def search_tweets(self, query, max_results=100):
        """
        Search for tweets matching a query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            pandas.DataFrame: DataFrame with tweet data
        """
        if not self.client:
            logger.warning("Twitter client not available. Cannot search tweets.")
            return pd.DataFrame()
            
        try:
            tweets = []
            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'lang', 'public_metrics']
            )
            
            if not response.data:
                logger.info(f"No tweets found for query: {query}")
                return pd.DataFrame()
                
            for tweet in response.data:
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'lang': tweet.lang,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'query': query
                }
                tweets.append(tweet_data)
                
            return pd.DataFrame(tweets)
        except Exception as e:
            logger.error(f"Error searching tweets for query {query}: {e}")
            return pd.DataFrame()
            
    def get_stock_tweets(self, symbol, max_results=100):
        """
        Get tweets related to a stock symbol.
        
        Args:
            symbol (str): Stock symbol
            max_results (int): Maximum number of results
            
        Returns:
            pandas.DataFrame: DataFrame with tweet data
        """
        query = f"${symbol} lang:en -is:retweet"
        logger.info(f"Searching tweets for: {query}")
        return self.search_tweets(query, max_results)
        
    def get_batch_tweets(self, symbols=None, max_results_per_symbol=50):
        """
        Get tweets for multiple stock symbols.
        
        Args:
            symbols (list): List of stock symbols
            max_results_per_symbol (int): Maximum results per symbol
            
        Returns:
            dict: Dictionary with tweet data for each symbol
        """
        symbols = symbols or config.STOCK_SYMBOLS
        result = {}
        
        for symbol in symbols:
            result[symbol] = self.get_stock_tweets(symbol, max_results_per_symbol)
            time.sleep(2)  # Avoid rate limiting
            
        return result

# Example usage
if __name__ == "__main__":
    if config.TWITTER_API_KEY != "YOUR_TWITTER_API_KEY":
        twitter_fetcher = TwitterDataFetcher()
        tweets_df = twitter_fetcher.get_stock_tweets("AAPL", 10)
        if not tweets_df.empty:
            print(f"Found {len(tweets_df)} tweets about $AAPL")
            print(tweets_df[['text', 'created_at', 'like_count']].head())
        else:
            print("No tweets found, or Twitter API is not configured correctly.")
    else:
        print("Set your Twitter API credentials in config.py to fetch tweets") 