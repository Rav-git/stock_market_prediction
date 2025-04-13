"""
Module for visualizing stock price predictions and sentiment analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class StockVisualizer:
    """Class for visualizing stock data and predictions."""
    
    def __init__(self, output_dir=None):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set up plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
    def plot_stock_prices(self, stock_data, title=None, save_path=None):
        """
        Plot historical stock prices.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert Date to datetime if it's not already
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Plot closing prices
        ax.plot(stock_data['Date'], stock_data['Close'], label='Close Price')
        
        # Add moving averages if they exist
        if 'MA5' in stock_data.columns:
            ax.plot(stock_data['Date'], stock_data['MA5'], label='5-day MA', linestyle='--')
        if 'MA20' in stock_data.columns:
            ax.plot(stock_data['Date'], stock_data['MA20'], label='20-day MA', linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title or f'Stock Price History for {stock_data["Symbol"].iloc[0]}')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_interactive_stock_prices(self, stock_data, title=None):
        """
        Create an interactive plot of stock prices using Plotly.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            title (str, optional): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Convert Date to datetime if it's not already
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Create a candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price'
        )])
        
        # Add moving averages if they exist
        if 'MA5' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['MA5'],
                mode='lines',
                name='5-day MA',
                line=dict(color='orange')
            ))
        
        if 'MA20' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='green')
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
            title=title or f'Interactive Stock Chart for {stock_data["Symbol"].iloc[0]}',
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
        
        return fig
    
    def plot_predictions(self, historical_data, predictions, title=None, save_path=None):
        """
        Plot historical prices with predictions.
        
        Args:
            historical_data (pandas.DataFrame): Historical stock data
            predictions (pandas.DataFrame): Predicted prices
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert dates to datetime
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        # Plot historical data
        ax.plot(historical_data['Date'], historical_data['Close'], label='Historical', color='blue')
        
        # Plot predictions
        ax.plot(predictions['Date'], predictions['Predicted_Close'], label='Predicted', color='red', linestyle='--')
        
        # Add a vertical line where historical data ends
        last_date = historical_data['Date'].iloc[-1]
        ax.axvline(x=last_date, color='black', linestyle='-', alpha=0.5)
        
        # Add text to indicate forecast starts
        ax.text(last_date, min(historical_data['Close']), 'Forecast', rotation=90, verticalalignment='bottom')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title or 'Stock Price Prediction')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_interactive_predictions(self, historical_data, predictions, title=None):
        """
        Create an interactive plot of predictions using Plotly.
        
        Args:
            historical_data (pandas.DataFrame): Historical stock data
            predictions (pandas.DataFrame): Predicted prices
            title (str, optional): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Convert dates to datetime
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted_Close'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8)
        ))
        
        # Add confidence interval if available
        if 'Upper_Bound' in predictions.columns and 'Lower_Bound' in predictions.columns:
            fig.add_trace(go.Scatter(
                x=predictions['Date'].tolist() + predictions['Date'].tolist()[::-1],
                y=predictions['Upper_Bound'].tolist() + predictions['Lower_Bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                name='Confidence Interval'
            ))
        
        # Customize layout
        fig.update_layout(
            title=title or 'Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            shapes=[
                # Add a vertical line at the transition point
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=historical_data['Date'].iloc[-1],
                    y0=0,
                    x1=historical_data['Date'].iloc[-1],
                    y1=1,
                    line=dict(
                        color="black",
                        width=1,
                        dash="dash",
                    ),
                )
            ],
            annotations=[
                dict(
                    x=historical_data['Date'].iloc[-1],
                    y=0.5,
                    xref="x",
                    yref="paper",
                    text="Forecast Start",
                    showarrow=True,
                    arrowhead=1,
                    ax=50,
                    ay=0
                )
            ]
        )
        
        return fig
    
    def plot_sentiment_over_time(self, sentiment_data, title=None, save_path=None):
        """
        Plot sentiment scores over time.
        
        Args:
            sentiment_data (pandas.DataFrame): Sentiment data
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if 'date' not in sentiment_data.columns or sentiment_data.empty:
            print("No valid sentiment data to plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert date to datetime if it's not already
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Sort by date
        sentiment_data = sentiment_data.sort_values('date')
        
        # Plot sentiment components
        ax.plot(sentiment_data['date'], sentiment_data['positive'], label='Positive', color='green')
        ax.plot(sentiment_data['date'], sentiment_data['negative'], label='Negative', color='red')
        ax.plot(sentiment_data['date'], sentiment_data['neutral'], label='Neutral', color='gray')
        ax.plot(sentiment_data['date'], sentiment_data['compound'], label='Compound', color='blue', linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.set_title(title or 'Sentiment Analysis Over Time')
        ax.legend()
        ax.grid(True)
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_interactive_sentiment(self, sentiment_data, stock_data=None, title=None):
        """
        Create an interactive plot of sentiment over time using Plotly.
        
        Args:
            sentiment_data (pandas.DataFrame): Sentiment data
            stock_data (pandas.DataFrame, optional): Stock data to overlay
            title (str, optional): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Check if sentiment data is available
        if sentiment_data is None or sentiment_data.empty:
            # Create an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No sentiment data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Sentiment Analysis - No Data Available")
            return fig
        
        # Convert date to datetime if it's not already
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add sentiment traces
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['compound'],
                mode='lines+markers',
                name='Compound Sentiment',
                line=dict(color='black', width=2),
                marker=dict(size=8)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['positive'],
                mode='lines',
                name='Positive',
                line=dict(color='green', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['negative'],
                mode='lines',
                name='Negative',
                line=dict(color='red', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['neutral'],
                mode='lines',
                name='Neutral',
                line=dict(color='blue', width=1.5)
            )
        )
        
        # Add stock price if provided
        if stock_data is not None and not stock_data.empty:
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Add stock price on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['Close'],
                    mode='lines',
                    name='Stock Price',
                    line=dict(color='orange', width=1.5, dash='dot'),
                    opacity=0.7
                ),
                secondary_y=True
            )
        
        # Add zero line for sentiment
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Customize layout
        fig.update_layout(
            title=title or 'Sentiment Analysis Over Time',
            xaxis_title='Date',
            yaxis_title='Sentiment',
            yaxis2_title='Stock Price',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
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
        
        return fig
    
    def create_sentiment_wordcloud(self, texts, title=None, save_path=None):
        """
        Create a word cloud from text data.
        
        Args:
            texts (list or pandas.Series): List of text data
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            from wordcloud import WordCloud
            import nltk
            from nltk.corpus import stopwords
            
            # Download stopwords if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            # Combine all texts
            if isinstance(texts, pd.Series):
                all_text = ' '.join(texts.dropna().astype(str))
            else:
                all_text = ' '.join([str(text) for text in texts if text])
                
            # Get stopwords
            stop_words = set(stopwords.words('english'))
            
            # Add common finance terms to stopwords
            finance_stop_words = ['stock', 'market', 'price', 'share', 'company', 'financial', 'investor']
            stop_words.update(finance_stop_words)
            
            # Create and generate a word cloud image
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                stopwords=stop_words,
                max_words=100,
                contour_width=3, 
                contour_color='steelblue'
            ).generate(all_text)
            
            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            
            if title:
                ax.set_title(title)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                
            return fig
            
        except ImportError:
            print("This function requires the wordcloud package. Please install it with 'pip install wordcloud'.")
            return None
    
    def plot_model_performance(self, actual, predicted, title=None, save_path=None):
        """
        Plot model performance (actual vs. predicted values).
        
        Args:
            actual (array-like): Actual values
            predicted (array-like): Predicted values
            title (str, optional): Plot title
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot actual vs. predicted
        ax1.scatter(actual, predicted, alpha=0.5)
        ax1.plot([min(actual), max(actual)], [min(actual), max(actual)], '--', color='red')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Actual vs. Predicted')
        ax1.grid(True)
        
        # Plot prediction errors
        errors = actual - predicted
        ax2.hist(errors, bins=20, alpha=0.7)
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.grid(True)
        
        plt.suptitle(title or 'Model Performance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    close_prices = np.sin(np.linspace(0, 10, 100)) * 10 + 100
    
    # Historical data
    historical_data = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.02,
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, 100),
        'Symbol': 'AAPL'
    })
    
    # Calculate moving averages
    historical_data['MA5'] = historical_data['Close'].rolling(window=5).mean()
    historical_data['MA20'] = historical_data['Close'].rolling(window=20).mean()
    
    # Predictions
    future_dates = pd.date_range(start='2022-04-11', periods=5, freq='D')
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': [105, 106, 107, 106, 105]
    })
    
    # Sentiment data
    sentiment_data = pd.DataFrame({
        'date': dates,
        'compound': np.sin(np.linspace(0, 15, 100)) * 0.5,
        'positive': np.abs(np.sin(np.linspace(0, 15, 100))) * 0.5,
        'negative': np.abs(np.cos(np.linspace(0, 15, 100))) * 0.3,
        'neutral': 0.2 + np.abs(np.sin(np.linspace(0, 7, 100))) * 0.2
    })
    
    # Create visualizer
    visualizer = StockVisualizer()
    
    # Plot stock prices
    fig1 = visualizer.plot_stock_prices(historical_data)
    plt.show()
    
    # Plot predictions
    fig2 = visualizer.plot_predictions(historical_data, predictions)
    plt.show()
    
    # Plot sentiment
    fig3 = visualizer.plot_sentiment_over_time(sentiment_data)
    plt.show() 