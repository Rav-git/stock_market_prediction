"""
Module for stock price prediction using machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import logging
import sys
import os
import joblib
from datetime import datetime, timedelta

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    """Stock price predictor using machine learning models."""
    
    def __init__(self, model_type='lstm'):
        """
        Initialize stock predictor.
        
        Args:
            model_type (str): Type of model to use ('lstm', 'rf', or 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 10  # For LSTM, number of time steps to look back
        self.prediction_horizon = config.PREDICTION_HORIZON
        self.sentiment_weight = config.SENTIMENT_WEIGHT
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _prepare_data(self, stock_data, sentiment_data=None):
        """
        Prepare data for training or prediction.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            sentiment_data (pandas.DataFrame, optional): Sentiment data
            
        Returns:
            tuple: Prepared data (features, targets, scaled data)
        """
        # Make a copy to avoid modifying the original
        stock_data = stock_data.copy()
        
        # Sort data by date
        stock_data = stock_data.sort_values('Date')
        
        # Create features for prediction
        stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['RSI'] = self._calculate_rsi(stock_data['Close'])
        stock_data['Price_Change'] = stock_data['Close'].pct_change()
        stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
        
        # Add day of week
        stock_data['Day_of_Week'] = pd.to_datetime(stock_data['Date']).dt.dayofweek
        
        # Add sentiment features if available
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_data = sentiment_data.sort_values('date')
            
            # Merge with stock data
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Use left join to keep all stock data
            stock_data = pd.merge(
                stock_data, 
                sentiment_data[['date', 'compound', 'positive', 'negative', 'neutral']],
                left_on='Date',
                right_on='date',
                how='left'
            )
            
            # Fill missing sentiment values with neutral
            stock_data['compound'] = stock_data['compound'].fillna(0)
            stock_data['positive'] = stock_data['positive'].fillna(0)
            stock_data['negative'] = stock_data['negative'].fillna(0)
            stock_data['neutral'] = stock_data['neutral'].fillna(1)
        else:
            # Add dummy sentiment columns if not provided
            stock_data['compound'] = 0
            stock_data['positive'] = 0
            stock_data['negative'] = 0
            stock_data['neutral'] = 1
            
        # Drop rows with NaN values
        stock_data = stock_data.dropna()
        
        # Select features for model
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 
                    'MA5', 'MA20', 'MA50', 'RSI', 'Price_Change', 'Volume_Change',
                    'compound', 'positive', 'negative', 'neutral', 'Day_of_Week']
                    
        # For LSTM model, we need to create sequences
        if self.model_type == 'lstm':
            return self._prepare_lstm_data(stock_data, features)
        else:
            # For other models
            X = stock_data[features].values
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create target variable (next day's closing price)
            y = stock_data['Close'].shift(-1).dropna().values
            
            # Remove the last row from X_scaled as it has no target
            X_scaled = X_scaled[:-1]
            
            return X_scaled, y, stock_data
    
    def _prepare_lstm_data(self, stock_data, features):
        """
        Prepare data specifically for LSTM model.
        
        Args:
            stock_data (pandas.DataFrame): Stock data
            features (list): Feature columns
            
        Returns:
            tuple: Prepared data for LSTM
        """
        # Ensure all features exist in the dataframe
        missing_features = [feat for feat in features if feat not in stock_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in stock_data: {missing_features}")
        
        # Get the features and scale them
        data = stock_data[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon):
            X.append(scaled_data[i:i + self.sequence_length])
            # Target is the closing price after prediction_horizon days
            y.append(scaled_data[i + self.sequence_length + self.prediction_horizon - 1, 0])  # 0 index is Close price
            
        return np.array(X), np.array(y), stock_data
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index.
        
        Args:
            prices (pandas.Series): Price series
            period (int): Period for RSI calculation
            
        Returns:
            pandas.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def build_model(self):
        """
        Build the prediction model.
        
        Returns:
            object: Built model
        """
        if self.model_type == 'lstm':
            return self._build_lstm_model()
        elif self.model_type == 'rf':
            return self._build_random_forest_model()
        elif self.model_type == 'ensemble':
            return self._build_ensemble_model()
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using LSTM.")
            return self._build_lstm_model()
    
    def _build_lstm_model(self):
        """
        Build LSTM model for time series prediction.
        
        Returns:
            tensorflow.keras.Model: LSTM model
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 16)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM model built successfully")
        return model
    
    def _build_random_forest_model(self):
        """
        Build Random Forest model for prediction.
        
        Returns:
            sklearn.ensemble.RandomForestRegressor: Random Forest model
        """
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        logger.info("Random Forest model built successfully")
        return model
    
    def _build_ensemble_model(self):
        """
        Build ensemble model (combination of models).
        
        Returns:
            dict: Dictionary with multiple models
        """
        # Build both LSTM and RF models
        lstm_model = self._build_lstm_model()
        rf_model = self._build_random_forest_model()
        
        return {
            'lstm': lstm_model,
            'rf': rf_model
        }
    
    def train(self, stock_data, sentiment_data=None):
        """
        Train the prediction model.
        
        Args:
            stock_data (pandas.DataFrame): Historical stock data
            sentiment_data (pandas.DataFrame, optional): Sentiment data
            
        Returns:
            object: Trained model
        """
        logger.info(f"Training {self.model_type} model...")
        
        try:
            # Make a copy of the data to avoid modifying the original
            stock_data_copy = stock_data.copy()
            
            # Ensure we have all required features by creating them
            # Create technical features
            stock_data_copy['MA5'] = stock_data_copy['Close'].rolling(window=5).mean()
            stock_data_copy['MA20'] = stock_data_copy['Close'].rolling(window=20).mean()
            stock_data_copy['MA50'] = stock_data_copy['Close'].rolling(window=50).mean()
            stock_data_copy['RSI'] = self._calculate_rsi(stock_data_copy['Close'])
            stock_data_copy['Price_Change'] = stock_data_copy['Close'].pct_change()
            stock_data_copy['Volume_Change'] = stock_data_copy['Volume'].pct_change()
            stock_data_copy['Day_of_Week'] = pd.to_datetime(stock_data_copy['Date']).dt.dayofweek
            
            # Add sentiment features if available or create dummy ones
            if sentiment_data is not None and not sentiment_data.empty:
                sentiment_data = sentiment_data.copy()
                sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
                stock_data_copy['Date'] = pd.to_datetime(stock_data_copy['Date'])
                
                stock_data_copy = pd.merge(
                    stock_data_copy, 
                    sentiment_data[['date', 'compound', 'positive', 'negative', 'neutral']],
                    left_on='Date',
                    right_on='date',
                    how='left'
                )
                stock_data_copy['compound'] = stock_data_copy['compound'].fillna(0)
                stock_data_copy['positive'] = stock_data_copy['positive'].fillna(0)
                stock_data_copy['negative'] = stock_data_copy['negative'].fillna(0)
                stock_data_copy['neutral'] = stock_data_copy['neutral'].fillna(1)
            else:
                stock_data_copy['compound'] = 0
                stock_data_copy['positive'] = 0
                stock_data_copy['negative'] = 0
                stock_data_copy['neutral'] = 1
            
            # Drop rows with NaN values (from rolling calculations)
            stock_data_copy = stock_data_copy.dropna()
            
            # Build the model based on model type
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model()
                X, y, _ = self._prepare_lstm_data(stock_data_copy, ['Close', 'Open', 'High', 'Low', 'Volume', 
                                                                'MA5', 'MA20', 'MA50', 'RSI', 'Price_Change', 'Volume_Change',
                                                                'compound', 'positive', 'negative', 'neutral', 'Day_of_Week'])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train LSTM model
                self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
                
            elif self.model_type == 'rf':
                self.model = self._build_random_forest_model()
                X, y, _ = self._prepare_data(stock_data_copy, sentiment_data)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train RF model
                self.model.fit(X_train, y_train)
                
            elif self.model_type == 'ensemble':
                self.model = self._build_ensemble_model()
                
                # Prepare data for LSTM
                X_lstm, y_lstm, _ = self._prepare_lstm_data(stock_data_copy, ['Close', 'Open', 'High', 'Low', 'Volume', 
                                                                         'MA5', 'MA20', 'MA50', 'RSI', 'Price_Change', 'Volume_Change',
                                                                         'compound', 'positive', 'negative', 'neutral', 'Day_of_Week'])
                
                # Split data for LSTM
                X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
                    X_lstm, y_lstm, test_size=0.2, random_state=42
                )
                
                # Train LSTM
                self.model['lstm'].fit(
                    X_lstm_train, y_lstm_train, 
                    epochs=100, batch_size=32, 
                    validation_data=(X_lstm_test, y_lstm_test), 
                    verbose=0
                )
                
                # Prepare data for RF
                X_rf, y_rf, _ = self._prepare_data(stock_data_copy, sentiment_data)
                
                # Split data for RF
                X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
                    X_rf, y_rf, test_size=0.2, random_state=42
                )
                
                # Train RF
                self.model['rf'].fit(X_rf_train, y_rf_train)
            
            # Save model
            self.save_model()
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            # If training fails, create a dummy model
            if self.model_type == 'ensemble':
                self.model = {
                    'lstm': None,
                    'rf': None
                }
            else:
                self.model = None
            return self.model
    
    def predict(self, stock_data, sentiment_data=None, days_ahead=5):
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
        
        try:
            # Prepare data for prediction
            X_data, _, prepared_data = self._prepare_data(stock_data, sentiment_data)
            
            # Last available date
            last_date = pd.to_datetime(stock_data['Date'].iloc[-1])
            
            # Create dates for prediction
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
            
            # Create dataframe for predictions
            predictions = pd.DataFrame({'Date': prediction_dates})
            
            # Predict based on model type
            if self.model_type == 'lstm':
                predicted_prices = self._predict_lstm(prepared_data, days_ahead)
            elif self.model_type == 'rf':
                predicted_prices = self._predict_rf(X_data, days_ahead)
            elif self.model_type == 'ensemble':
                predicted_prices = self._predict_ensemble(prepared_data, X_data, days_ahead)
            else:
                # Default to a simple prediction if unknown model type
                logger.warning(f"Unknown model type {self.model_type}. Using simple prediction.")
                # Simple prediction based on the last 5 days trend
                last_price = stock_data['Close'].iloc[-1]
                last_5_days_change = stock_data['Close'].pct_change().tail(5).mean()
                predicted_prices = [last_price * (1 + last_5_days_change * (i+1)) for i in range(days_ahead)]
            
            # Add predictions to dataframe
            predictions['Predicted_Close'] = predicted_prices
            
            # Add confidence bounds (approximated)
            volatility = stock_data['Close'].pct_change().std()
            confidence_level = 1.96  # 95% confidence level
            
            predictions['Upper_Bound'] = predictions['Predicted_Close'] * (1 + volatility * confidence_level)
            predictions['Lower_Bound'] = predictions['Predicted_Close'] * (1 - volatility * confidence_level)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Return a simple prediction as fallback
            last_date = pd.to_datetime(stock_data['Date'].iloc[-1])
            last_price = stock_data['Close'].iloc[-1]
            
            # Create dates for prediction
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
            
            # Simple moving average prediction
            last_5_days_change = stock_data['Close'].pct_change().tail(5).mean()
            predicted_prices = [last_price * (1 + last_5_days_change * (i+1)) for i in range(days_ahead)]
            
            # Create dataframe for predictions
            predictions = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted_Close': predicted_prices
            })
            
            # Add confidence bounds (approximated)
            volatility = stock_data['Close'].pct_change().std()
            confidence_level = 1.96  # 95% confidence level
            
            predictions['Upper_Bound'] = predictions['Predicted_Close'] * (1 + volatility * confidence_level)
            predictions['Lower_Bound'] = predictions['Predicted_Close'] * (1 - volatility * confidence_level)
            
            return predictions
    
    def _predict_lstm(self, prepared_data, days_ahead):
        """
        Predict using LSTM model.
        
        Args:
            prepared_data (pandas.DataFrame): Prepared dataframe
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            list: Predicted prices
        """
        # Ensure we have the LSTM model
        if 'lstm' not in self.model:
            # Simple predictions if model not available
            last_price = prepared_data['Close'].iloc[-1]
            last_5_days_change = prepared_data['Close'].pct_change().tail(5).mean()
            return [last_price * (1 + last_5_days_change * (i+1)) for i in range(days_ahead)]
            
        # Get features and create sequences for LSTM
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 
                    'MA5', 'MA20', 'MA50', 'RSI', 'Price_Change', 'Volume_Change',
                    'compound', 'positive', 'negative', 'neutral', 'Day_of_Week']
                    
        # Get the scaled data
        data = prepared_data[features].values
        scaled_data = self.scaler.transform(data)
        
        # Use the last sequence for prediction
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(features))
        
        # Predict next step
        first_prediction_scaled = self.model['lstm'].predict(last_sequence)[0]
        
        # Convert prediction back to original scale (focusing on Close price)
        # Create a dummy array with the right shape
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = first_prediction_scaled  # Set the Close price
        
        # Inverse transform
        predicted_price = self.scaler.inverse_transform(dummy)[0, 0]
        
        # Create result list with first prediction
        predicted_prices = [predicted_price]
        
        # Since we can't build proper sequences for future timepoints without features,
        # we'll use a simple trend-based prediction for the remaining days
        last_5_days_change = prepared_data['Close'].pct_change().tail(5).mean()
        for i in range(1, days_ahead):
            next_price = predicted_prices[-1] * (1 + last_5_days_change)
            predicted_prices.append(next_price)
        
        return predicted_prices
    
    def _predict_rf(self, X_data, days_ahead):
        """
        Predict using Random Forest model.
        
        Args:
            X_data (numpy.ndarray): Prepared data for Random Forest
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            list: Predicted prices
        """
        # If we don't have a model or it's not in the right format, return simple predictions
        if self.model is None or (isinstance(self.model, dict) and 'rf' not in self.model):
            # Get last closing price from scaler to inverse transform
            last_price = self.scaler.inverse_transform(X_data[-1:])[:, 0][0]
            # Simple trend-based prediction
            last_5_data_points = X_data[-5:]
            avg_change = np.mean(np.diff(last_5_data_points[:, 0])) 
            return [last_price * (1 + avg_change * (i+1)) for i in range(days_ahead)]
            
        # For RF, use the last data point
        last_data_point = X_data[-1:]
        
        # Get the model, handle both single model and ensemble dict
        rf_model = self.model if not isinstance(self.model, dict) else self.model.get('rf')
        
        # Predict first price
        predicted_price = rf_model.predict(last_data_point)[0]
        
        # Initialize predictions list
        predicted_prices = [predicted_price]
        
        # For RF, we'll use a more simple approach for future predictions
        # based on the first prediction and the recent trend
        last_5_data_points = X_data[-5:]
        avg_change = np.mean(np.diff(last_5_data_points[:, 0])) 
        
        for i in range(1, days_ahead):
            next_price = predicted_prices[0] * (1 + avg_change * i)
            predicted_prices.append(next_price)
        
        return predicted_prices
    
    def _predict_ensemble(self, prepared_data, X_data, days_ahead):
        """
        Predict using ensemble model.
        
        Args:
            prepared_data (pandas.DataFrame): Prepared data for LSTM
            X_data (numpy.ndarray): Prepared data for Random Forest
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            list: Predicted prices
        """
        # If we don't have a model or it's not a dict with the right keys, return simple predictions
        if self.model is None or not isinstance(self.model, dict) or 'rf' not in self.model or 'lstm' not in self.model:
            # Get last closing price 
            last_price = prepared_data['Close'].iloc[-1]
            # Simple trend-based prediction
            last_5_days_change = prepared_data['Close'].pct_change().tail(5).mean()
            return [last_price * (1 + last_5_days_change * (i+1)) for i in range(days_ahead)]
            
        # Get LSTM prediction
        lstm_predictions = self._predict_lstm(prepared_data, days_ahead)
        
        # Get RF prediction
        rf_predictions = self._predict_rf(X_data, days_ahead)
        
        # Combine predictions (simple average)
        ensemble_predictions = [(lstm + rf) / 2 for lstm, rf in zip(lstm_predictions, rf_predictions)]
        
        return ensemble_predictions
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_type}_model_{timestamp}"
        
        try:
            if self.model_type == 'lstm':
                model_path = os.path.join(self.models_dir, f"{filename}.h5")
                self.model.save(model_path)
                
                # Save scaler separately
                scaler_path = os.path.join(self.models_dir, f"{filename}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                
            elif self.model_type == 'rf':
                model_path = os.path.join(self.models_dir, f"{filename}.pkl")
                joblib.dump(self.model, model_path)
                
                # Save scaler separately
                scaler_path = os.path.join(self.models_dir, f"{filename}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                
            elif self.model_type == 'ensemble':
                # Save LSTM model
                lstm_path = os.path.join(self.models_dir, f"{filename}_lstm.h5")
                self.model['lstm'].save(lstm_path)
                
                # Save RF model
                rf_path = os.path.join(self.models_dir, f"{filename}_rf.pkl")
                joblib.dump(self.model['rf'], rf_path)
                
                # Save scaler
                scaler_path = os.path.join(self.models_dir, f"{filename}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
                
            logger.info(f"Model saved successfully: {filename}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str, optional): Path to the saved scaler
            
        Returns:
            object: Loaded model
        """
        try:
            if self.model_type == 'lstm':
                self.model = tf.keras.models.load_model(model_path)
            elif self.model_type == 'rf':
                self.model = joblib.load(model_path)
            elif self.model_type == 'ensemble':
                # If model_path is a directory containing both models
                if os.path.isdir(model_path):
                    lstm_path = os.path.join(model_path, 'lstm.h5')
                    rf_path = os.path.join(model_path, 'rf.pkl')
                    
                    self.model = {
                        'lstm': tf.keras.models.load_model(lstm_path),
                        'rf': joblib.load(rf_path)
                    }
                else:
                    # If paths to individual models are provided in model_path as a dict
                    self.model = {
                        'lstm': tf.keras.models.load_model(model_path['lstm']),
                        'rf': joblib.load(model_path['rf'])
                    }
            
            # Load scaler if provided
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            logger.info(f"Model loaded successfully from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Create some sample data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    close_prices = np.sin(np.linspace(0, 10, 100)) * 10 + 100
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.02,
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    # Create a predictor and train it
    predictor = StockPredictor(model_type='rf')
    predictor.train(data)
    
    # Make predictions
    predictions = predictor.predict(data, days_ahead=5)
    print("Predictions:")
    print(predictions) 