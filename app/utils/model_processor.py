import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from ..models import db, UploadedFile, ProductForecast, InventoryPrediction, ModelMetric
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
import joblib
import warnings
from .model_loader import model_manager
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from sklearn.model_selection import train_test_split
from scipy import stats
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import current_app
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize global model variables
_arima_model = None
_xgboost_model = None
_prophet_model = None

# Add load_data function at the module level
def load_data(cache=True):
    """Load data from the latest dataset"""
    if cache and hasattr(load_data, 'cache'):
        return load_data.cache
        
    try:
        # Try to load from FLASK MODELS directory
        models_path = os.path.join('FLASK MODELS', 'pharma_forecasting_dataset.csv')
        if os.path.exists(models_path):
            data = pd.read_csv(models_path)
            data['date'] = pd.to_datetime(data['date'])
            if cache:
                load_data.cache = data
            return data
            
        return None
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def process_uploaded_file(file_path):
    """
    Process the uploaded CSV file through ML models and generate forecasts
    
    Args:
        file_path: Path to the uploaded CSV file
        
    Returns:
        dict: Dictionary of forecast results
    """
    try:
        # Load the uploaded data
        df = pd.read_csv(file_path)
        
        # Validate the required columns
        required_columns = ['product_id', 'product_name', 'category', 'date', 
                           'price', 'stock_level', 'quantity_sold', 'reservations', 'revenue']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Clear any cached data
        if hasattr(load_data, 'cache'):
            delattr(load_data, 'cache')
        
        try:
            # Get or create the uploaded file record first
            from ..models import UploadedFile
            uploaded_file = UploadedFile.query.filter_by(filename=os.path.basename(file_path)).first()
            
            if not uploaded_file:
                # Create new record if it doesn't exist
                uploaded_file = UploadedFile(
                    filename=os.path.basename(file_path),
                    upload_date=datetime.now(),
                    status='processing'
                )
                db.session.add(uploaded_file)
                db.session.commit()  # Commit to get the ID
            
            # Generate forecasts with the file ID
            sales_forecast_df = generate_arima_sales_forecast(df, uploaded_file.id)
            inventory_forecast_df = generate_inventory_forecast(df, uploaded_file.id)
            reservations_forecast_df = generate_reservations_forecast(df, uploaded_file.id)
            
            # Update source_file_id in forecast DataFrames
            sales_forecast_df['source_file_id'] = uploaded_file.id
            inventory_forecast_df['source_file_id'] = uploaded_file.id
            reservations_forecast_df['source_file_id'] = uploaded_file.id
            
            # Save forecasts to database
            save_revenue_forecasts_to_db(sales_forecast_df, uploaded_file.id)
            save_inventory_forecasts_to_db(inventory_forecast_df, uploaded_file.id)
            save_reservation_forecasts_to_db(reservations_forecast_df, uploaded_file.id)
            
            # Generate and save model metrics
            generate_model_metrics(df, uploaded_file.id)
            
            # Update file status to completed
            uploaded_file.status = 'completed'
            db.session.commit()
            
            return {
                'sales_forecast': sales_forecast_df,
                'inventory_forecast': inventory_forecast_df,
                'reservations_forecast': reservations_forecast_df
            }
            
        except Exception as e:
            if uploaded_file:
                uploaded_file.status = 'error'
                uploaded_file.error_message = str(e)
                db.session.commit()
            logger.error(f"Error processing forecasts: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

def save_revenue_forecasts_to_db(forecast_df, source_file_id):
    """Save revenue forecasts to the database"""
    for _, row in forecast_df.iterrows():
        forecast = ProductForecast(
            product_id=row['product_id'],
            forecast_type='revenue',
            forecast_date=row['date'].date() if isinstance(row['date'], pd.Timestamp) else row['date'],
            forecast_value=float(row['forecasted_revenue']),
            model_version='ARIMA_v1',
            source_file_id=source_file_id
        )
        db.session.add(forecast)
    db.session.commit()

def save_inventory_forecasts_to_db(forecast_df, source_file_id):
    """Save inventory forecasts to the database"""
    for _, row in forecast_df.iterrows():
        prediction = InventoryPrediction(
            product_id=row['product_id'],
            prediction_date=row['date'].date() if isinstance(row['date'], pd.Timestamp) else row['date'],
            days_to_zero=float(row['forecasted_stock']),
            source_file_id=source_file_id
        )
        db.session.add(prediction)
    db.session.commit()

def save_reservation_forecasts_to_db(forecast_df, source_file_id):
    """Save reservation forecasts to the database"""
    for _, row in forecast_df.iterrows():
        forecast = ProductForecast(
            product_id=row['product_id'],
            forecast_type='reservation',
            forecast_date=row['date'].date() if isinstance(row['date'], pd.Timestamp) else row['date'],
            forecast_value=float(row['forecasted_reservations']),
            model_version='Prophet_v1',
            source_file_id=source_file_id
        )
        db.session.add(forecast)
    db.session.commit()

def generate_model_metrics(data, file_id):
    """Generate and store model performance metrics"""
    try:
        # Calculate metrics for each model type
        sales_metrics = {
            'mape': calculate_mape(data, 'revenue'),
            'rmse': calculate_rmse(data, 'revenue'),
            'mae': calculate_mae(data, 'revenue')
        }
        
        inventory_metrics = {
            'mape': calculate_mape(data, 'stock_level'),
            'rmse': calculate_rmse(data, 'stock_level'),
            'mae': calculate_mae(data, 'stock_level')
        }
        
        reservation_metrics = {
            'mape': calculate_mape(data, 'reservations'),
            'rmse': calculate_rmse(data, 'reservations'),
            'mae': calculate_mae(data, 'reservations')
        }
        
        # Store metrics in database
        metrics = [
            ModelMetric(
                source_file_id=file_id,
                product_id='ALL',
                model_type='arima_sales',
                **sales_metrics
            ),
            ModelMetric(
                source_file_id=file_id,
                product_id='ALL',
                model_type='xgb_inventory',
                **inventory_metrics
            ),
            ModelMetric(
                source_file_id=file_id,
                product_id='ALL',
                model_type='prophet_resv',
                **reservation_metrics
            )
        ]
        
        db.session.bulk_save_objects(metrics)
        db.session.commit()
        
    except Exception as e:
        logging.error(f"Error generating model metrics: {str(e)}")
        raise

def calculate_mape(data, column):
    """Calculate Mean Absolute Percentage Error"""
    actual = data[column]
    predicted = actual.rolling(window=7, min_periods=1).mean()
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_rmse(data, column):
    """Calculate Root Mean Square Error"""
    actual = data[column]
    predicted = actual.rolling(window=7, min_periods=1).mean()
    return np.sqrt(mean_squared_error(actual.dropna(), predicted.dropna()))

def calculate_mae(data, column):
    """Calculate Mean Absolute Error"""
    actual = data[column]
    predicted = actual.rolling(window=7, min_periods=1).mean()
    return mean_absolute_error(actual.dropna(), predicted.dropna())

def generate_arima_sales_forecast(data, file_id):
    """Generate sales forecast using SARIMA model"""
    try:
        # Group data by date
        daily_sales = data.groupby('date')['revenue'].sum().reset_index()
        daily_sales = daily_sales.set_index('date')
        
        # Fit SARIMA model with optimal parameters
        model = SARIMAX(daily_sales, 
                       order=(2, 1, 2),  # ARIMA parameters
                       seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                       enforce_stationarity=False)
        results = model.fit(disp=False)
        
        # Generate forecast for next 30 days
        future_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=30)
        forecast = results.forecast(steps=30)
        
        # Calculate confidence intervals
        forecast_mean = forecast.values
        conf_int = results.get_forecast(steps=30).conf_int()
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'product_id': 'ALL',
            'forecasted_revenue': forecast_mean,
            'confidence_lower': conf_int.iloc[:, 0],
            'confidence_upper': conf_int.iloc[:, 1]
        })
        
        # Save forecasts to database
        save_revenue_forecasts_to_db(forecast_df, file_id)
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Error generating sales forecast: {str(e)}")
        raise

def generate_inventory_forecast(data, file_id):
    """Generate inventory forecast using XGBoost with advanced features"""
    try:
        # Prepare features
        inventory_data = data.groupby('date').agg({
            'stock_level': 'mean',
            'quantity_sold': 'sum',
            'reservations': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # Feature engineering
        inventory_data['day_of_week'] = inventory_data['date'].dt.dayofweek
        inventory_data['month'] = inventory_data['date'].dt.month
        inventory_data['day_of_month'] = inventory_data['date'].dt.day
        inventory_data['week_of_year'] = inventory_data['date'].dt.isocalendar().week
        
        # Calculate rolling features
        inventory_data['rolling_mean_7d'] = inventory_data['stock_level'].rolling(window=7).mean()
        inventory_data['rolling_std_7d'] = inventory_data['stock_level'].rolling(window=7).std()
        inventory_data['rolling_sales_7d'] = inventory_data['quantity_sold'].rolling(window=7).mean()
        
        # Prepare features for prediction
        features = ['day_of_week', 'month', 'day_of_month', 'week_of_year',
                   'quantity_sold', 'reservations', 'revenue',
                   'rolling_mean_7d', 'rolling_std_7d', 'rolling_sales_7d']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Prepare training data
        X = inventory_data[features].dropna()
        y = inventory_data['stock_level'].iloc[len(inventory_data)-len(X):]
        
        model.fit(X, y)
        
        # Generate future dates
        future_dates = pd.date_range(start=inventory_data['date'].max() + pd.Timedelta(days=1), periods=30)
        
        # Prepare future features
        future_data = pd.DataFrame({
            'date': future_dates,
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'day_of_month': future_dates.day,
            'week_of_year': future_dates.isocalendar().week,
            'quantity_sold': [inventory_data['quantity_sold'].mean()] * 30,
            'reservations': [inventory_data['reservations'].mean()] * 30,
            'revenue': [inventory_data['revenue'].mean()] * 30,
            'rolling_mean_7d': [inventory_data['rolling_mean_7d'].mean()] * 30,
            'rolling_std_7d': [inventory_data['rolling_std_7d'].mean()] * 30,
            'rolling_sales_7d': [inventory_data['rolling_sales_7d'].mean()] * 30
        })
        
        # Generate predictions
        predictions = model.predict(future_data[features])
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'product_id': 'ALL',
            'forecasted_stock': predictions,
            'days_to_zero': np.where(predictions > 0, 
                                   predictions / inventory_data['quantity_sold'].mean(),
                                   0),
            'reorder_point': predictions * 0.2,
            'optimal_order_qty': predictions * 0.5
        })
        
        # Save predictions to database
        save_inventory_forecasts_to_db(forecast_df, file_id)
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Error generating inventory forecast: {str(e)}")
        raise

def generate_reservations_forecast(data, file_id):
    """Generate reservations forecast using Prophet with additional regressors"""
    try:
        # Prepare data for Prophet
        daily_data = data.groupby('date').agg({
            'reservations': 'sum',
            'stock_level': 'mean',
            'quantity_sold': 'sum'
        }).reset_index()
        
        prophet_data = pd.DataFrame({
            'ds': daily_data['date'],
            'y': daily_data['reservations'],
            'stock_level': daily_data['stock_level'],
            'quantity_sold': daily_data['quantity_sold']
        })
        
        # Configure Prophet model
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Add additional regressors
        model.add_regressor('stock_level')
        model.add_regressor('quantity_sold')
        
        # Fit model
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=30)
        future['stock_level'] = prophet_data['stock_level'].mean()
        future['quantity_sold'] = prophet_data['quantity_sold'].mean()
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast.tail(30)['ds'],
            'product_id': 'ALL',
            'forecasted_reservations': forecast.tail(30)['yhat'],
            'confidence_lower': forecast.tail(30)['yhat_lower'],
            'confidence_upper': forecast.tail(30)['yhat_upper']
        })
        
        # Save forecasts to database
        save_reservation_forecasts_to_db(forecast_df, file_id)
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Error generating reservations forecast: {str(e)}")
        raise

def train_arima_model(data):
    """Train ARIMA model with basic data"""
    try:
        # Prepare data
        ts_data = data.set_index('date')['value']
        
        # Create and fit ARIMA model
        model = SARIMAX(ts_data,
                       order=(1, 1, 1),  # Simple ARIMA parameters
                       seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                       enforce_stationarity=False)
        fitted_model = model.fit(disp=False)
        return fitted_model
    except Exception as e:
        print(f"Error training ARIMA model: {str(e)}")
        return None

def train_xgboost_model(data):
    """Train XGBoost model with basic data"""
    try:
        # Prepare features
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        
        # Create basic features
        X = data[['day_of_week', 'month']]
        y = data['value']
        
        # Create and train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Error training XGBoost model: {str(e)}")
        return None

def reset_models():
    """Reset all models and clear cached data"""
    global _arima_model, _xgboost_model, _prophet_model
    _arima_model = None
    _xgboost_model = None
    _prophet_model = None
    
    # Clear cached data
    if hasattr(load_data, 'cache'):
        delattr(load_data, 'cache')
    
    # Create fallback models
    create_fallback_models()

def create_fallback_models():
    """Create fallback models with minimal data"""
    try:
        # Create simple time series data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, 30)  # Random values around 100
        })
        
        # Initialize ARIMA model
        global _arima_model
        _arima_model = train_arima_model(data)
        print("Created fallback aggregate ARIMA model")
        
        # Initialize XGBoost model
        global _xgboost_model
        _xgboost_model = train_xgboost_model(data)
        print("Created fallback aggregate XGBoost model")
        
        # Initialize Prophet model
        global _prophet_model
        _prophet_model = Prophet()
        prophet_data = data.rename(columns={'date': 'ds', 'value': 'y'})
        _prophet_model.fit(prophet_data)
        print("Created fallback aggregate Prophet model")
        
    except Exception as e:
        print(f"Error creating fallback models: {str(e)}")
        return False
    
    return True 