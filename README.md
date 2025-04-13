# Stock Market Prediction

A comprehensive system for stock market prediction integrating technical analysis with sentiment analysis. This project uses machine learning techniques to predict stock price movements and provides an interactive visualization dashboard.

## Features

- **Data Collection**: Fetch historical stock data using Yahoo Finance API and financial news using NewsAPI
- **Sentiment Analysis**: Analyze financial news sentiment using VADER, TextBlob, and ensemble techniques
- **Price Prediction**: Predict stock prices using LSTM, Random Forest, and ensemble models
- **Interactive Dashboard**: Visualize stock data, technical indicators, predictions, and sentiment trends
- **Stock Comparison**: Compare multiple stocks with normalized prices and correlation metrics

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Rav-git/stock_market_prediction.git
   cd stock_market_prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Copy `config.py.example` to `config.py`
   - Add your API keys for Yahoo Finance, NewsAPI, etc.

### Usage

1. Run the Streamlit web application:
   ```
   python -m streamlit run src/app_simple.py
   ```

2. Run batch analysis:
   ```
   python src/main.py
   ```

## Project Structure

- `src/` - Source code
  - `app_simple.py` - Streamlit web application
  - `main.py` - Command-line batch analysis
  - `data_collection/` - Stock data and news fetching modules
  - `sentiment_analysis/` - Sentiment analysis modules
  - `prediction/` - Stock prediction models
  - `visualization/` - Data visualization utilities

- `models/` - Pre-trained models and model storage
- `data/` - Raw and processed data storage
- `output/` - Output files and visualizations
- `figures/` - Diagram descriptions for the research paper
- `Stock_Market_Prediction_Research_Paper.tex` - Research paper documenting the approach

## Acknowledgments

- [Yahoo Finance API](https://pypi.org/project/yfinance/) for stock data
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Streamlit](https://streamlit.io/) for the interactive dashboard
- 
