"""
Main application for stock market prediction and sentiment analysis.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import modules
from data_collection.stock_data import StockDataFetcher, FinancialNewsFetcher
from data_collection.social_data import TwitterDataFetcher
from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from prediction.stock_predictor import StockPredictor
from visualization.visualizer import StockVisualizer

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_sentiment.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockSentimentApp:
    """Main application class."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the application.
        
        Args:
            output_dir (str, optional): Directory to save outputs
        """
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize components
        self.stock_fetcher = StockDataFetcher()
        self.news_fetcher = FinancialNewsFetcher()
        self.twitter_fetcher = TwitterDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer(method='ensemble')
        self.predictor = StockPredictor(model_type='ensemble')
        self.visualizer = StockVisualizer(output_dir=self.output_dir)
        
        logger.info("StockSentimentApp initialized")
        
    def collect_data(self, symbol, period="1y", days_for_news=30):
        """
        Collect all data for a stock.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period for historical data
            days_for_news (int): Days of news to fetch
            
        Returns:
            tuple: Stock data, news data, and social media data
        """
        logger.info(f"Collecting data for {symbol}")
        
        # Get stock data
        stock_data = self.stock_fetcher.get_historical_data(symbol, period=period)
        
        # Initialize empty placeholders for news and social data
        news_data = []
        social_data = pd.DataFrame()
        
        # Only try to get news data if the API key is properly set
        if config.NEWS_API_KEY != "YOUR_NEWS_API_KEY":
            try:
                news_data = self.news_fetcher.get_news_for_symbol(symbol, days=days_for_news)
                logger.info(f"Fetched {len(news_data)} news articles")
            except Exception as e:
                logger.warning(f"Could not fetch news data: {e}")
        else:
            logger.warning("News API key not set. Skipping news data.")
        
        # Only try to get social data if the API keys are properly set
        if config.TWITTER_API_KEY != "YOUR_TWITTER_API_KEY":
            try:
                social_data = self.twitter_fetcher.get_stock_tweets(symbol, max_results=100)
                logger.info(f"Fetched {len(social_data)} tweets")
            except Exception as e:
                logger.warning(f"Could not fetch social data: {e}")
        else:
            logger.warning("Twitter API keys not set. Skipping social data.")
        
        return stock_data, news_data, social_data
    
    def analyze_sentiment(self, news_data, social_data):
        """
        Analyze sentiment from news and social media.
        
        Args:
            news_data (list): News articles
            social_data (pandas.DataFrame): Social media posts
            
        Returns:
            tuple: Analyzed news and social data
        """
        logger.info("Analyzing sentiment")
        
        # Analyze news
        news_sentiment = self.sentiment_analyzer.analyze_news(news_data)
        
        # Analyze social media
        social_sentiment = self.sentiment_analyzer.analyze_tweets(social_data)
        
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
    
    def train_model(self, stock_data, sentiment_data=None):
        """
        Train the prediction model.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            sentiment_data (pandas.DataFrame, optional): Sentiment data
            
        Returns:
            object: Trained model
        """
        logger.info("Training prediction model")
        return self.predictor.train(stock_data, sentiment_data)
    
    def make_predictions(self, stock_data, sentiment_data=None, days_ahead=5):
        """
        Make price predictions.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            sentiment_data (pandas.DataFrame, optional): Sentiment data
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            pandas.DataFrame: Predictions
        """
        logger.info(f"Making predictions for {days_ahead} days ahead")
        return self.predictor.predict(stock_data, sentiment_data, days_ahead)
    
    def visualize_results(self, stock_data, sentiment_data, predictions, symbol):
        """
        Create visualizations of results.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            sentiment_data (pandas.DataFrame): Sentiment data
            predictions (pandas.DataFrame): Price predictions
            symbol (str): Stock symbol
        """
        logger.info("Creating visualizations")
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot stock prices
        fig1 = self.visualizer.plot_stock_prices(
            stock_data,
            title=f'Stock Price History for {symbol}',
            save_path=os.path.join(self.output_dir, f'{symbol}_price_history_{timestamp}.png')
        )
        
        # Plot interactive stock prices
        fig2 = self.visualizer.plot_interactive_stock_prices(
            stock_data,
            title=f'Interactive Stock Chart for {symbol}'
        )
        
        # Save interactive chart as HTML
        fig2.write_html(os.path.join(self.output_dir, f'{symbol}_interactive_chart_{timestamp}.html'))
        
        # Plot price predictions
        fig3 = self.visualizer.plot_predictions(
            stock_data,
            predictions,
            title=f'Stock Price Prediction for {symbol}',
            save_path=os.path.join(self.output_dir, f'{symbol}_prediction_{timestamp}.png')
        )
        
        # Plot interactive predictions
        fig4 = self.visualizer.plot_interactive_predictions(
            stock_data,
            predictions,
            title=f'Interactive Price Prediction for {symbol}'
        )
        
        # Save interactive predictions as HTML
        fig4.write_html(os.path.join(self.output_dir, f'{symbol}_interactive_prediction_{timestamp}.html'))
        
        # Plot sentiment if available
        if sentiment_data is not None and not sentiment_data.empty:
            fig5 = self.visualizer.plot_sentiment_over_time(
                sentiment_data,
                title=f'Sentiment Analysis for {symbol}',
                save_path=os.path.join(self.output_dir, f'{symbol}_sentiment_{timestamp}.png')
            )
            
            # Plot interactive sentiment
            fig6 = self.visualizer.plot_interactive_sentiment(
                sentiment_data,
                stock_data,
                title=f'Interactive Sentiment Analysis for {symbol}'
            )
            
            # Save interactive sentiment as HTML
            fig6.write_html(os.path.join(self.output_dir, f'{symbol}_interactive_sentiment_{timestamp}.html'))
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def run(self, symbol=None, period="1y", days_for_news=30, days_ahead=5, train=True):
        """
        Run the full workflow for a stock.
        
        Args:
            symbol (str, optional): Stock symbol to analyze
            period (str): Time period for historical data
            days_for_news (int): Days of news to fetch
            days_ahead (int): Number of days to predict ahead
            train (bool): Whether to train a new model
            
        Returns:
            dict: Results including predictions and sentiment
        """
        symbol = symbol or config.STOCK_SYMBOLS[0]
        logger.info(f"Running full workflow for {symbol}")
        
        # Collect data
        stock_data, news_data, social_data = self.collect_data(symbol, period, days_for_news)
        
        if not stock_data.empty:
            logger.info(f"Successfully collected stock data for {symbol} with {len(stock_data)} rows")
        else:
            logger.error(f"Failed to collect stock data for {symbol}")
            return None
        
        # Analyze sentiment if we have news or social data
        sentiment_by_date = None
        news_sentiment = None
        social_sentiment = None
        
        if news_data or not social_data.empty:
            news_sentiment, social_sentiment, sentiment_by_date = self.analyze_sentiment(news_data, social_data)
            logger.info("Sentiment analysis completed")
        else:
            logger.warning("No news or social data available for sentiment analysis")
        
        # Train model if requested
        if train:
            self.train_model(stock_data, sentiment_by_date)
        
        # Make predictions
        predictions = self.make_predictions(stock_data, sentiment_by_date, days_ahead)
        
        # Visualize results
        self.visualize_results(stock_data, sentiment_by_date, predictions, symbol)
        
        # Generate summary
        latest_sentiment = None
        if sentiment_by_date is not None and not sentiment_by_date.empty:
            latest_sentiment = sentiment_by_date.iloc[-1].to_dict()
        
        current_price = stock_data['Close'].iloc[-1]
        predicted_price = predictions['Predicted_Close'].iloc[-1]
        price_change = ((predicted_price / current_price) - 1) * 100
        
        summary = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_percent': price_change,
            'prediction_direction': 'up' if price_change > 0 else 'down',
            'latest_sentiment': latest_sentiment,
            'prediction_date': predictions['Date'].iloc[-1].strftime('%Y-%m-%d')
        }
        
        # Save summary to file
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(self.output_dir, f'{symbol}_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)
        
        return {
            'stock_data': stock_data,
            'sentiment_data': sentiment_by_date,
            'predictions': predictions,
            'summary': summary
        }
    
    def run_batch(self, symbols=None, period="1y", days_for_news=30, days_ahead=5):
        """
        Run the workflow for multiple stocks.
        
        Args:
            symbols (list, optional): List of stock symbols
            period (str): Time period for historical data
            days_for_news (int): Days of news to fetch
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            dict: Results for each symbol
        """
        symbols = symbols or config.STOCK_SYMBOLS
        logger.info(f"Running batch analysis for {len(symbols)} stocks")
        
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.run(symbol, period, days_for_news, days_ahead)
                # Sleep to avoid API rate limits
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Generate combined summary
        summaries = [results[symbol]['summary'] for symbol in results]
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(os.path.join(self.output_dir, f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), index=False)
        
        return results

def main():
    """Command line interface for the application."""
    parser = argparse.ArgumentParser(description='Stock Market Prediction with Sentiment Analysis')
    parser.add_argument('--symbol', type=str, help='Stock symbol to analyze')
    parser.add_argument('--period', type=str, default='1y', help='Time period for historical data')
    parser.add_argument('--news-days', type=int, default=30, help='Days of news to fetch')
    parser.add_argument('--predict-days', type=int, default=5, help='Days to predict ahead')
    parser.add_argument('--batch', action='store_true', help='Run batch analysis for all symbols in config')
    parser.add_argument('--no-train', action='store_true', help='Skip model training')
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs')
    
    args = parser.parse_args()
    
    app = StockSentimentApp(output_dir=args.output_dir)
    
    if args.batch:
        app.run_batch(period=args.period, days_for_news=args.news_days, days_ahead=args.predict_days)
    else:
        app.run(symbol=args.symbol, period=args.period, days_for_news=args.news_days, days_ahead=args.predict_days, train=not args.no_train)
    
    logger.info("Application completed successfully")

if __name__ == "__main__":
    main() 