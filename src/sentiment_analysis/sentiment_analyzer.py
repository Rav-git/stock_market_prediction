"""
Module for sentiment analysis of financial news and social media data.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer
import logging
import sys
import os
import re
# Comment out transformers import to avoid TensorFlow errors
# from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if not already done
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    
class SentimentAnalyzer:
    """Class for analyzing sentiment in text data."""
    
    def __init__(self, method='ensemble'):
        """
        Initialize sentiment analyzer.
        
        Args:
            method (str): Sentiment analysis method ('vader', 'textblob', or 'ensemble')
        """
        # Force method to be 'ensemble_no_transformers' to avoid TensorFlow errors
        if method == 'transformers' or method == 'ensemble':
            logger.info("Transformers model disabled. Using ensemble without transformers.")
            self.method = 'ensemble_no_transformers'
        else:
            self.method = method
            
        self.vader = VaderSentimentIntensityAnalyzer()
        self.nltk_sid = SentimentIntensityAnalyzer()
        
        # Disable transformers model loading
        self.transformers_model = None
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'sentiment': 'neutral'
            }
            
        text = self.preprocess_text(text)
        
        if self.method == 'vader':
            return self._analyze_vader(text)
        elif self.method == 'textblob':
            return self._analyze_textblob(text)
        elif self.method == 'transformers' and self.transformers_model:
            # This will not be reached since we disabled transformers
            return self._analyze_vader(text)
        elif self.method == 'ensemble':
            # This will not be reached since we force 'ensemble_no_transformers'
            return self._analyze_ensemble_no_transformers(text)
        elif self.method == 'ensemble_no_transformers':
            return self._analyze_ensemble_no_transformers(text)
        else:
            # Default to VADER
            return self._analyze_vader(text)
    
    def _analyze_vader(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        scores = self.vader.polarity_scores(text)
        sentiment = 'neutral'
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
            
        scores['sentiment'] = sentiment
        return scores
    
    def _analyze_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Convert TextBlob polarity to VADER-like scores for consistency
        if polarity > 0:
            positive = polarity
            negative = 0
            neutral = 1 - positive
        elif polarity < 0:
            positive = 0
            negative = -polarity
            neutral = 1 - negative
        else:
            positive = 0
            negative = 0
            neutral = 1
            
        compound = polarity  # TextBlob polarity as compound
        
        sentiment = 'neutral'
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
            
        return {
            'compound': compound,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'sentiment': sentiment
        }
    
    def _analyze_transformers(self, text):
        """
        Analyze sentiment using transformers.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        try:
            # Truncate text if it's too long for the model
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
                
            result = self.transformers_model(text)[0]
            label = result['label']
            score = result['score']
            
            # Convert FinBERT labels to consistent format
            if label == 'positive':
                sentiment = 'positive'
                compound = score
                positive = score
                negative = 0
                neutral = 1 - score
            elif label == 'negative':
                sentiment = 'negative'
                compound = -score
                positive = 0
                negative = score
                neutral = 1 - score
            else:  # neutral
                sentiment = 'neutral'
                compound = 0
                positive = 0
                negative = 0
                neutral = score
                
            return {
                'compound': compound,
                'positive': positive,
                'neutral': neutral,
                'negative': negative,
                'sentiment': sentiment
            }
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {e}")
            # Fall back to VADER
            return self._analyze_vader(text)
    
    def _analyze_ensemble(self, text):
        """
        Analyze sentiment using an ensemble of methods.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        # Get scores from different methods
        vader_scores = self._analyze_vader(text)
        textblob_scores = self._analyze_textblob(text)
        
        if self.transformers_model:
            transformer_scores = self._analyze_transformers(text)
            
            # Combine scores (weighted average)
            compound = (
                vader_scores['compound'] * 0.4 +
                textblob_scores['compound'] * 0.3 +
                transformer_scores['compound'] * 0.3
            )
            
            positive = (
                vader_scores['positive'] * 0.4 +
                textblob_scores['positive'] * 0.3 +
                transformer_scores['positive'] * 0.3
            )
            
            negative = (
                vader_scores['negative'] * 0.4 +
                textblob_scores['negative'] * 0.3 +
                transformer_scores['negative'] * 0.3
            )
            
            neutral = (
                vader_scores['neutral'] * 0.4 +
                textblob_scores['neutral'] * 0.3 +
                transformer_scores['neutral'] * 0.3
            )
        else:
            # Without transformers, just use VADER and TextBlob
            compound = vader_scores['compound'] * 0.6 + textblob_scores['compound'] * 0.4
            positive = vader_scores['positive'] * 0.6 + textblob_scores['positive'] * 0.4
            negative = vader_scores['negative'] * 0.6 + textblob_scores['negative'] * 0.4
            neutral = vader_scores['neutral'] * 0.6 + textblob_scores['neutral'] * 0.4
            
        # Determine overall sentiment
        sentiment = 'neutral'
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
            
        return {
            'compound': compound,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'sentiment': sentiment
        }
        
    def _analyze_ensemble_no_transformers(self, text):
        """
        Analyze sentiment using an ensemble without transformers.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        # Get scores from different methods
        vader_scores = self._analyze_vader(text)
        textblob_scores = self._analyze_textblob(text)
        
        # Combine scores (weighted average)
        compound = vader_scores['compound'] * 0.6 + textblob_scores['compound'] * 0.4
        positive = vader_scores['positive'] * 0.6 + textblob_scores['positive'] * 0.4
        negative = vader_scores['negative'] * 0.6 + textblob_scores['negative'] * 0.4
        neutral = vader_scores['neutral'] * 0.6 + textblob_scores['neutral'] * 0.4
        
        # Determine overall sentiment
        sentiment = 'neutral'
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
            
        return {
            'compound': compound,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'sentiment': sentiment
        }
    
    def analyze_dataframe(self, df, text_column):
        """
        Analyze sentiment for a DataFrame with text data.
        
        Args:
            df (pandas.DataFrame): DataFrame with text data
            text_column (str): Column name containing text
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment scores
        """
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or {text_column} column not found")
            return df
            
        results = []
        for _, row in df.iterrows():
            text = row[text_column]
            sentiment = self.analyze_text(text)
            row_dict = row.to_dict()
            row_dict.update(sentiment)
            results.append(row_dict)
            
        return pd.DataFrame(results)
    
    def analyze_news(self, news_data):
        """
        Analyze sentiment for news articles.
        
        Args:
            news_data (list): List of news articles
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment scores
        """
        if not news_data:
            logger.warning("No news data provided")
            return pd.DataFrame()
            
        df = pd.DataFrame(news_data)
        # Combine title and description for better sentiment analysis
        df['content'] = df['title'] + ". " + df['description'].fillna('')
        
        return self.analyze_dataframe(df, 'content')
    
    def analyze_tweets(self, tweets_df):
        """
        Analyze sentiment for tweets.
        
        Args:
            tweets_df (pandas.DataFrame): DataFrame with tweets
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment scores
        """
        return self.analyze_dataframe(tweets_df, 'text')
    
    def get_overall_sentiment(self, df):
        """
        Get overall sentiment from analyzed data.
        
        Args:
            df (pandas.DataFrame): DataFrame with sentiment scores
            
        Returns:
            dict: Overall sentiment scores
        """
        if df.empty or 'compound' not in df.columns:
            logger.warning("DataFrame is empty or missing sentiment scores")
            return {
                'compound': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'sentiment': 'neutral'
            }
            
        # Calculate weighted average based on metrics like retweets, likes, etc.
        if 'retweet_count' in df.columns and 'like_count' in df.columns:
            # For tweets, weight by engagement
            weights = df['retweet_count'] + df['like_count'] + 1  # Add 1 to avoid zero weights
        else:
            # Equal weights
            weights = pd.Series([1] * len(df))
            
        compound = (df['compound'] * weights).sum() / weights.sum()
        positive = (df['positive'] * weights).sum() / weights.sum()
        negative = (df['negative'] * weights).sum() / weights.sum()
        neutral = (df['neutral'] * weights).sum() / weights.sum()
        
        sentiment = 'neutral'
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
            
        return {
            'compound': compound,
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'sentiment': sentiment
        }

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(method='ensemble')
    
    # Example with a single text
    text = "Tesla's new electric car model exceeds all expectations and sales forecasts."
    sentiment = analyzer.analyze_text(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment['sentiment']}, Compound score: {sentiment['compound']:.2f}")
    
    # Example with a list of texts
    texts = [
        "Apple's quarterly earnings report shows record profits.",
        "Microsoft stock plummets after disappointing product launch.",
        "Google's new AI technology receives mixed reviews from experts."
    ]
    
    for text in texts:
        sentiment = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment['sentiment']}, Compound score: {sentiment['compound']:.2f}") 