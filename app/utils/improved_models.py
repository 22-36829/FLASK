import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ImprovedForecaster:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
    def preprocess_data(self, data, target_col):
        """
        Preprocess data with advanced techniques for better accuracy
        """
        # Convert to datetime if needed
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Add time-based features
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        
        # Add rolling statistics
        data[f'{target_col}_7d_ma'] = data[target_col].rolling(window=7, min_periods=1).mean()
        data[f'{target_col}_30d_ma'] = data[target_col].rolling(window=30, min_periods=1).mean()
        
        return data
        
    def train_product_sales_model(self, data, product_id):
        """
        Train SARIMAX model for product sales with optimized parameters
        """
        try:
            # Filter data for specific product
            product_data = data[data['product_id'] == product_id].copy()
            
            # Preprocess
            processed_data = self.preprocess_data(product_data, 'quantity_sold')
            
            # SARIMAX parameters optimized for pharmaceutical sales - simplified for stability
            model = SARIMAX(processed_data['quantity_sold'],
                          exog=processed_data[['price', 'stock_level', 'reservations']],
                          order=(1, 1, 1),           # Simplified ARIMA parameters
                          seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                          enforce_stationarity=False)
            
            fitted_model = model.fit(disp=False, maxiter=50)  # Limit iterations
            
            # Store model
            self.models[f'sales_{product_id}'] = {
                'model': fitted_model,
                'last_date': processed_data.index[-1],
                'exog_columns': ['price', 'stock_level', 'reservations']
            }
            
            # Calculate accuracy metrics
            y_pred = fitted_model.get_prediction(start=processed_data.index[0])
            y_true = processed_data['quantity_sold']
            
            mae = mean_absolute_error(y_true, y_pred.predicted_mean)
            mse = mean_squared_error(y_true, y_pred.predicted_mean)
            r2 = r2_score(y_true, y_pred.predicted_mean)
            
            accuracy = 1 - (mae / y_true.mean())  # Normalized accuracy
            
            self.metrics[f'sales_{product_id}'] = {
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'accuracy': accuracy * 100  # Convert to percentage
            }
            
            logger.info(f"Trained sales model for {product_id} with accuracy: {accuracy * 100:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error training sales model for {product_id}: {str(e)}")
            return False
            
    def train_inventory_model(self, data, product_id):
        """
        Train SARIMAX model for inventory prediction
        """
        try:
            # Filter data for specific product
            product_data = data[data['product_id'] == product_id].copy()
            
            # Preprocess
            processed_data = self.preprocess_data(product_data, 'stock_level')
            
            # SARIMAX parameters optimized for inventory - simplified for stability
            model = SARIMAX(processed_data['stock_level'],
                          exog=processed_data[['quantity_sold', 'stock_in', 'stock_out']],
                          order=(1, 1, 1),           # Simplified ARIMA parameters
                          seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                          enforce_stationarity=False)
            
            fitted_model = model.fit(disp=False, maxiter=50)  # Limit iterations
            
            # Store model
            self.models[f'inventory_{product_id}'] = {
                'model': fitted_model,
                'last_date': processed_data.index[-1],
                'exog_columns': ['quantity_sold', 'stock_in', 'stock_out']
            }
            
            # Calculate accuracy metrics
            y_pred = fitted_model.get_prediction(start=processed_data.index[0])
            y_true = processed_data['stock_level']
            
            mae = mean_absolute_error(y_true, y_pred.predicted_mean)
            mse = mean_squared_error(y_true, y_pred.predicted_mean)
            r2 = r2_score(y_true, y_pred.predicted_mean)
            
            accuracy = 1 - (mae / y_true.mean())  # Normalized accuracy
            
            self.metrics[f'inventory_{product_id}'] = {
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'accuracy': accuracy * 100  # Convert to percentage
            }
            
            logger.info(f"Trained inventory model for {product_id} with accuracy: {accuracy * 100:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error training inventory model for {product_id}: {str(e)}")
            return False
            
    def train_revenue_model(self, data, product_id):
        """
        Train SARIMAX model for revenue forecasting
        """
        try:
            # Filter data for specific product
            product_data = data[data['product_id'] == product_id].copy()
            
            # Preprocess
            processed_data = self.preprocess_data(product_data, 'revenue')
            
            # SARIMAX parameters optimized for revenue - simplified for stability
            model = SARIMAX(processed_data['revenue'],
                          exog=processed_data[['quantity_sold', 'price']],
                          order=(1, 1, 1),           # Simplified ARIMA parameters
                          seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                          enforce_stationarity=False)
            
            fitted_model = model.fit(disp=False, maxiter=50)  # Limit iterations
            
            # Store model
            self.models[f'revenue_{product_id}'] = {
                'model': fitted_model,
                'last_date': processed_data.index[-1],
                'exog_columns': ['quantity_sold', 'price']
            }
            
            # Calculate accuracy metrics
            y_pred = fitted_model.get_prediction(start=processed_data.index[0])
            y_true = processed_data['revenue']
            
            mae = mean_absolute_error(y_true, y_pred.predicted_mean)
            mse = mean_squared_error(y_true, y_pred.predicted_mean)
            r2 = r2_score(y_true, y_pred.predicted_mean)
            
            accuracy = 1 - (mae / y_true.mean())  # Normalized accuracy
            
            self.metrics[f'revenue_{product_id}'] = {
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'accuracy': accuracy * 100  # Convert to percentage
            }
            
            logger.info(f"Trained revenue model for {product_id} with accuracy: {accuracy * 100:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error training revenue model for {product_id}: {str(e)}")
            return False
    
    def predict_next_n_days(self, product_id, n_days=30, forecast_type='sales'):
        """
        Generate predictions for the next n days
        """
        try:
            model_key = f'{forecast_type}_{product_id}'
            if model_key not in self.models:
                raise ValueError(f"No trained model found for {model_key}")
                
            model_info = self.models[model_key]
            model = model_info['model']
            last_date = model_info['last_date']
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days)
            
            # For demonstration, we'll use the last known exog values
            # In production, you might want to forecast these values as well
            last_exog = pd.DataFrame(index=future_dates)
            for col in model_info['exog_columns']:
                last_exog[col] = model.data.exog[-1, model_info['exog_columns'].index(col)]
            
            # Generate forecast
            forecast = model.get_forecast(steps=n_days, exog=last_exog[model_info['exog_columns']])
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast.predicted_mean,
                'lower_ci': forecast.conf_int()[:, 0],
                'upper_ci': forecast.conf_int()[:, 1]
            })
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return None

# Example usage:
# forecaster = ImprovedForecaster()
# forecaster.train_product_sales_model(data, 'P001')
# forecaster.train_inventory_model(data, 'P001')
# forecaster.train_revenue_model(data, 'P001')
# 
# # Get predictions
# sales_forecast = forecaster.predict_next_n_days('P001', forecast_type='sales')
# inventory_forecast = forecaster.predict_next_n_days('P001', forecast_type='inventory')
# revenue_forecast = forecaster.predict_next_n_days('P001', forecast_type='revenue') 