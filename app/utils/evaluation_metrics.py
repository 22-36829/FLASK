import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

logger = logging.getLogger(__name__)

def calculate_growth_rate(data):
    """Calculate growth rate from the first to last month"""
    try:
        # Group data by month for growth rate calculation
        monthly_sales = data.groupby(pd.Grouper(key='date', freq='M'))['revenue'].sum()
        
        if len(monthly_sales) >= 2:
            first_month_sales = monthly_sales.iloc[0]
            last_month_sales = monthly_sales.iloc[-1]
            growth_rate = ((last_month_sales - first_month_sales) / first_month_sales) * 100
            return f"{growth_rate:.1f}"
        return "0.0"
    except Exception as e:
        print(f"Error calculating growth rate: {str(e)}")
        return "N/A"

def find_peak_month(data):
    """Find the month with highest sales"""
    try:
        monthly_sales = data.groupby(pd.Grouper(key='date', freq='M'))['revenue'].sum()
        peak_month = monthly_sales.idxmax().strftime('%B %Y')
        return peak_month
    except Exception as e:
        print(f"Error finding peak month: {str(e)}")
        return 'N/A'

def detect_seasonality(data):
    """Detect seasonality pattern in sales data"""
    try:
        # Resample to daily frequency and fill missing values
        daily_sales = data.groupby('date')['revenue'].sum()
        daily_sales = daily_sales.reindex(
            pd.date_range(daily_sales.index.min(), 
                         daily_sales.index.max(), 
                         freq='D')
        ).fillna(daily_sales.mean())
        
        # Only perform seasonal decomposition if we have enough data points
        if len(daily_sales) >= 14:  # Need at least 2 weeks of data
            # Use 7-day period for weekly seasonality
            result = seasonal_decompose(daily_sales, period=7, model='additive')
            seasonal_strength = np.std(result.seasonal) / np.std(daily_sales)
            return "Strong" if seasonal_strength > 0.5 else "Moderate" if seasonal_strength > 0.3 else "Weak"
        return "Insufficient data"
    except Exception as e:
        print(f"Error detecting seasonality: {str(e)}")
        return "Unable to determine"

def calculate_revenue_metrics(data, validation_split=0.2):
    """Calculate improved revenue metrics using SARIMA model"""
    try:
        if data.empty:
            return {'mape': 'N/A', 'r2': 'N/A', 'rmse': 'N/A'}
            
        # Prepare time series data
        daily_revenue = data.groupby('date')['revenue'].sum().reset_index()
        daily_revenue = daily_revenue.set_index('date')
        
        # Split data into train and validation
        train_size = int(len(daily_revenue) * (1 - validation_split))
        train = daily_revenue[:train_size]
        test = daily_revenue[train_size:]
        
        # Fit SARIMA model with optimal parameters
        model = SARIMAX(train, 
                       order=(2, 1, 2),  # ARIMA parameters
                       seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                       enforce_stationarity=False)
        results = model.fit(disp=False)
        
        # Make predictions
        predictions = results.predict(start=test.index[0], end=test.index[-1])
        
        # Calculate metrics
        mape = np.mean(np.abs((test['revenue'] - predictions) / test['revenue'])) * 100
        r2 = r2_score(test['revenue'], predictions)
        rmse = np.sqrt(mean_squared_error(test['revenue'], predictions))
        
        return {
            'mape': round(mape, 2),
            'r2': round(r2, 2),
            'rmse': round(rmse, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating revenue metrics: {str(e)}")
        return {'mape': 'N/A', 'r2': 'N/A', 'rmse': 'N/A'}

def calculate_inventory_metrics(data, validation_split=0.2):
    """Calculate improved inventory metrics using XGBoost with feature engineering"""
    try:
        if data.empty:
            return {'mae': 'N/A', 'accuracy': 'N/A'}
            
        # Feature engineering
        inventory_data = data.groupby('date').agg({
            'stock_level': 'mean',
            'quantity_sold': 'sum',
            'reservations': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # Create additional features
        inventory_data['day_of_week'] = inventory_data['date'].dt.dayofweek
        inventory_data['month'] = inventory_data['date'].dt.month
        inventory_data['day_of_month'] = inventory_data['date'].dt.day
        inventory_data['week_of_year'] = inventory_data['date'].dt.isocalendar().week
        
        # Calculate rolling features
        inventory_data['rolling_mean_7d'] = inventory_data['stock_level'].rolling(window=7).mean()
        inventory_data['rolling_std_7d'] = inventory_data['stock_level'].rolling(window=7).std()
        inventory_data['rolling_sales_7d'] = inventory_data['quantity_sold'].rolling(window=7).mean()
        
        # Drop NaN values after creating rolling features
        inventory_data = inventory_data.dropna()
        
        # Prepare features and target
        features = ['day_of_week', 'month', 'day_of_month', 'week_of_year',
                   'quantity_sold', 'reservations', 'revenue',
                   'rolling_mean_7d', 'rolling_std_7d', 'rolling_sales_7d']
        X = inventory_data[features]
        y = inventory_data['stock_level']
        
        # Split data
        train_size = int(len(inventory_data) * (1 - validation_split))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost with optimized parameters
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        accuracy = 100 * (1 - np.mean(np.abs((y_test - predictions) / y_test)))
        
        return {
            'mae': round(mae, 2),
            'accuracy': round(accuracy, 1)
        }
    except Exception as e:
        logger.error(f"Error calculating inventory metrics: {str(e)}")
        return {'mae': 'N/A', 'accuracy': 'N/A'}

def calculate_reservation_metrics(data, validation_split=0.2):
    """Calculate improved reservation metrics using Prophet with additional regressors"""
    try:
        if data.empty:
            return {'rmse': 'N/A', 'precision': 'N/A'}
            
        # Prepare data for Prophet
        daily_reservations = data.groupby('date').agg({
            'reservations': 'sum',
            'stock_level': 'mean',
            'quantity_sold': 'sum'
        }).reset_index()
        
        prophet_data = pd.DataFrame({
            'ds': daily_reservations['date'],
            'y': daily_reservations['reservations'],
            'stock_level': daily_reservations['stock_level'],
            'quantity_sold': daily_reservations['quantity_sold']
        })
        
        # Split data
        train_size = int(len(prophet_data) * (1 - validation_split))
        train = prophet_data[:train_size]
        test = prophet_data[train_size:]
        
        if len(test) < 2:  # Need at least 2 points for validation
            return {'rmse': 'N/A', 'precision': 'N/A'}
        
        # Configure and train Prophet model
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
        model.fit(train)
        
        # Make predictions
        future = test.copy()
        forecast = model.predict(future)
        
        # Calculate metrics
        y_true = test['y'].values
        y_pred = forecast['yhat'].values
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate precision more robustly
        try:
            # Calculate absolute percentage errors
            abs_errors = np.abs(y_true - y_pred)
            actual_values = np.abs(y_true)
            
            # Handle zero values in actual_values
            mask = actual_values != 0
            if not mask.any():  # If all actual values are zero
                precision = 0.0
            else:
                # Calculate percentage errors only for non-zero actual values
                percentage_errors = abs_errors[mask] / actual_values[mask]
                # Calculate precision as 1 - MAPE (mean absolute percentage error)
                precision = (1 - np.mean(percentage_errors)) * 100
                # Clip precision to 0-100 range
                precision = np.clip(precision, 0, 100)
        except Exception as e:
            logger.error(f"Error calculating precision: {str(e)}")
            precision = 0.0
        
        return {
            'rmse': round(rmse, 2),
            'precision': round(precision, 1)
        }
    except Exception as e:
        logger.error(f"Error calculating reservation metrics: {str(e)}")
        return {'rmse': 'N/A', 'precision': 'N/A'}

def calculate_price_metrics(data):
    """Calculate price-related metrics"""
    try:
        if data.empty:
            return {'volatility': 'N/A', 'margin_trend': 'N/A'}
            
        # Calculate daily average prices
        daily_prices = data.groupby('date')['price'].mean().reset_index()
        
        # Calculate price volatility
        price_changes = daily_prices['price'].pct_change()
        volatility = price_changes.std() * 100
        
        # Calculate margin trend
        data['margin'] = (data['price'] - data['unit_cost']) / data['price'] * 100
        start_margin = data.groupby(data['date'])['margin'].mean().iloc[0]
        end_margin = data.groupby(data['date'])['margin'].mean().iloc[-1]
        margin_trend = end_margin - start_margin
        
        return {
            'volatility': round(volatility, 1),
            'margin_trend': round(margin_trend, 1)
        }
    except Exception as e:
        logger.error(f"Error calculating price metrics: {str(e)}")
        return {'volatility': 'N/A', 'margin_trend': 'N/A'}

def calculate_sales_metrics(data):
    """Calculate sales performance metrics"""
    try:
        if data.empty:
            return {'growth_rate': 'N/A', 'peak_month': 'N/A', 'seasonality': 'N/A'}
            
        # Calculate daily sales
        daily_sales = data.groupby('date')['revenue'].sum().reset_index()
        
        # Calculate growth rate
        start_value = daily_sales['revenue'].iloc[0]
        end_value = daily_sales['revenue'].iloc[-1]
        growth_rate = ((end_value - start_value) / start_value) * 100
        
        # Find peak month
        monthly_sales = data.groupby(data['date'].dt.strftime('%B'))['revenue'].sum()
        peak_month = monthly_sales.idxmax()
        
        # Calculate seasonality strength
        decomposition = seasonal_decompose(daily_sales.set_index('date')['revenue'], period=30)
        seasonality = np.std(decomposition.seasonal) / np.std(decomposition.resid)
        
        return {
            'growth_rate': round(growth_rate, 1),
            'peak_month': peak_month,
            'seasonality': round(seasonality, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating sales metrics: {str(e)}")
        return {'growth_rate': 'N/A', 'peak_month': 'N/A', 'seasonality': 'N/A'} 