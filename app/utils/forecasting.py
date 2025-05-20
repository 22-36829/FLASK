import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from .model_evaluation import calculate_forecast_metrics, plot_forecast_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def prepare_time_series_data(data, target_col, date_col='date', product_id=None):
    """Prepare data for time series analysis"""
    if product_id:
        data = data[data['product_id'] == product_id]
    
    # Ensure date column is datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Aggregate by date
    ts_data = data.groupby(date_col)[target_col].sum().reset_index()
    ts_data = ts_data.sort_values(date_col)
    
    return ts_data

def find_optimal_arima_order(data, max_p=3, max_d=2, max_q=3):
    """Find optimal ARIMA parameters using AIC"""
    best_aic = float('inf')
    best_order = None
    
    # First, determine optimal d using Augmented Dickey-Fuller test
    adf_result = adfuller(data)
    optimal_d = 0
    if adf_result[1] > 0.05:  # If p-value > 0.05, series is non-stationary
        optimal_d = 1
        # Test one more time after differencing
        diff_data = np.diff(data)
        adf_result = adfuller(diff_data)
        if adf_result[1] > 0.05:
            optimal_d = 2
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(data, order=(p, optimal_d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, optimal_d, q)
            except:
                continue
    
    return best_order or (1, optimal_d, 1)

def train_arima_model(data, target_col, date_col='date', product_id=None):
    """Train ARIMA model for sales forecasting"""
    try:
        # Prepare data
        ts_data = prepare_time_series_data(data, target_col, date_col, product_id)
        
        if len(ts_data) < 10:
            raise ValueError("Insufficient data points for ARIMA modeling")
            
        # Split data into train and test
        train_size = int(len(ts_data) * 0.8)
        train = ts_data[:train_size]
        test = ts_data[train_size:]
        
        # Fit ARIMA model
        model = ARIMA(train[target_col], order=(1,1,1))
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.forecast(steps=len(test))
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(test[target_col], predictions),
            'rmse': np.sqrt(mean_squared_error(test[target_col], predictions)),
            'mae': mean_absolute_error(test[target_col], predictions),
            'r2': r2_score(test[target_col], predictions),
            'mape': np.mean(np.abs((test[target_col] - predictions) / test[target_col])) * 100
        }
        
        return model_fit, predictions, metrics, test
        
    except Exception as e:
        print(f"Error in ARIMA model: {str(e)}")
        return None, None, None, None

def train_prophet_model(data, target_col, date_col='date', product_id=None):
    """Train Prophet model for reservation forecasting"""
    try:
        # Prepare data with normalization
        ts_data, scaler = prepare_time_series_data(data, target_col, date_col, product_id, normalize=True)
        
        if len(ts_data) < 10:
            return None
        
        # Prepare data for Prophet
        prophet_data = ts_data.rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Split data
        train_size = int(len(prophet_data) * 0.8)
        train = prophet_data.iloc[:train_size]
        test = prophet_data.iloc[train_size:]
        
        if len(test) == 0:
            return None
        
        # Configure Prophet model with simpler parameters
        model = Prophet(
            yearly_seasonality=False,  # Simplified seasonality
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,  # More rigid trend
            seasonality_prior_scale=1.0,   # Less flexible seasonality
            changepoint_range=0.8
        )
        
        # Fit the model
        model.fit(train)
        
        # Make predictions
        future = model.make_future_dataframe(
            periods=len(test),
            freq='D',
            include_history=False
        )
        
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
        
        # Denormalize predictions and actuals if normalized
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(test['y'].values.reshape(-1, 1)).flatten()
        else:
            actuals = test['y'].values
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(actuals, predictions, 'Prophet')
        
        # Create evaluation plot
        eval_plot = plot_forecast_evaluation(
            actuals,
            predictions,
            test['ds'].values,
            f'Prophet {target_col.replace("_", " ").title()} Forecast Evaluation'
        )
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': metrics,
            'evaluation_plot': eval_plot
        }
    
    except Exception as e:
        print(f"Error in Prophet model: {str(e)}")
        return None

def prepare_inventory_features(data, product_id=None):
    """Prepare features for inventory prediction"""
    try:
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        if product_id:
            df = df[df['product_id'] == product_id]
        
        # Validate required columns
        required_cols = ['date', 'quantity_sold', 'stock_level', 'reservations', 'price', 'revenue']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date to datetime if needed
        df['date'] = pd.to_datetime(df['date'])
        
        # Initial data cleaning and type conversion
        numeric_cols = ['quantity_sold', 'stock_level', 'reservations', 'price', 'revenue']
        for col in numeric_cols:
            # Convert to numeric, coerce errors to NaN, then fill NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Ensure non-negative values
            df[col] = df[col].clip(lower=0)
        
        # Extract date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Create lag features with different windows
        for lag in [1, 3, 7]:
            # Sales lags
            df[f'sales_lag_{lag}'] = df.groupby('product_id')['quantity_sold'].shift(lag).fillna(0)
            df[f'sales_ma_{lag}'] = df.groupby('product_id')['quantity_sold'].transform(
                lambda x: x.rolling(window=lag, min_periods=1).mean()
            ).fillna(0)
            
            # Reservation lags
            df[f'reservations_lag_{lag}'] = df.groupby('product_id')['reservations'].shift(lag).fillna(0)
            df[f'reservations_ma_{lag}'] = df.groupby('product_id')['reservations'].transform(
                lambda x: x.rolling(window=lag, min_periods=1).mean()
            ).fillna(0)
            
            # Stock level lags
            df[f'stock_lag_{lag}'] = df.groupby('product_id')['stock_level'].shift(lag).fillna(0)
            df[f'stock_ma_{lag}'] = df.groupby('product_id')['stock_level'].transform(
                lambda x: x.rolling(window=lag, min_periods=1).mean()
            ).fillna(0)
        
        # Improved safe rate of change calculation
        def safe_pct_change(series):
            series = pd.to_numeric(series, errors='coerce')
            diff = series.diff()
            prev = series.shift(1)
            # Handle division by zero and infinite values
            mask = (prev != 0) & (np.abs(diff) < np.inf)
            result = pd.Series(np.zeros(len(series)), index=series.index)
            result[mask] = (diff[mask] / prev[mask])
            # Clip extreme values
            return result.clip(-1, 1)
        
        df['sales_rate'] = df.groupby('product_id')['quantity_sold'].transform(safe_pct_change)
        df['stock_rate'] = df.groupby('product_id')['stock_level'].transform(safe_pct_change)
        df['reservation_rate'] = df.groupby('product_id')['reservations'].transform(safe_pct_change)
        
        # Improved ratio features with safe division
        def safe_ratio(numerator, denominator, fill_value=0, max_value=10):
            numerator = pd.to_numeric(numerator, errors='coerce')
            denominator = pd.to_numeric(denominator, errors='coerce')
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(denominator != 0, numerator / denominator, fill_value)
                return np.clip(ratio, 0, max_value)
        
        df['sales_to_stock_ratio'] = safe_ratio(df['quantity_sold'], df['stock_level'])
        df['reservations_to_stock_ratio'] = safe_ratio(df['reservations'], df['stock_level'])
        
        # Add moving averages for stability
        df['sales_ma_30'] = df.groupby('product_id')['quantity_sold'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        ).fillna(0)
        
        df['stock_ma_30'] = df.groupby('product_id')['stock_level'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        ).fillna(0)
        
        # Winsorize numeric features to handle outliers
        def winsorize(series, limits=(0.05, 0.95)):
            series = pd.to_numeric(series, errors='coerce')
            if series.nunique() <= 1:
                return series
            lower = series.quantile(limits[0])
            upper = series.quantile(limits[1])
            return series.clip(lower=lower, upper=upper)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in ['day_of_week', 'month', 'day_of_month', 'quarter', 
                          'is_weekend', 'is_month_start', 'is_month_end']:
                df[col] = df.groupby('product_id')[col].transform(winsorize)
        
        # Final validation and cleanup
        # Convert any remaining non-numeric values to 0
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Replace any remaining infinite values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        # Verify no NaN or infinite values remain
        assert not df.isnull().any().any(), "NaN values found in processed data"
        assert not np.isinf(df.values).any(), "Infinite values found in processed data"
        
        return df
        
    except Exception as e:
        print(f"Error in prepare_inventory_features: {str(e)}")
        return None

def train_xgboost_model(data, product_id=None):
    """Train XGBoost model for inventory prediction"""
    try:
        # Prepare features
        features = prepare_inventory_features(data, product_id)
        
        if features is None:
            print("Feature preparation failed")
            return None
        
        if len(features) < 10:
            print("Insufficient data points for training")
            return None
        
        # Define feature columns
        feature_cols = [
            # Basic features
            'quantity_sold', 'reservations', 'revenue', 'price',
            'day_of_week', 'month', 'day_of_month', 'quarter',
            'is_weekend', 'is_month_start', 'is_month_end',
            
            # Lag features
            'sales_lag_1', 'sales_lag_3', 'sales_lag_7',
            'reservations_lag_1', 'reservations_lag_3', 'reservations_lag_7',
            'stock_lag_1', 'stock_lag_3', 'stock_lag_7',
            
            # Moving averages
            'sales_ma_7', 'sales_ma_30',
            'reservations_ma_7', 'reservations_ma_30',
            'stock_ma_7', 'stock_ma_30',
            
            # Rate features
            'sales_rate', 'stock_rate', 'reservation_rate',
            
            # Ratio features
            'sales_to_stock_ratio', 'reservations_to_stock_ratio'
        ]
        
        # Validate all features exist
        missing_features = [col for col in feature_cols if col not in features.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None
        
        X = features[feature_cols]
        y = features['stock_level']
        
        # Validate no NaN or infinite values
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("Data contains NaN or infinite values after feature selection")
            return None
        
        # Split data chronologically
        train_size = int(len(X) * 0.8)
        if train_size < 5:  # Require at least 5 training samples
            print("Insufficient training data")
            return None
            
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        scaler = RobustScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Error in feature scaling: {str(e)}")
            return None
        
        # Train XGBoost model with conservative parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,  # Even more conservative
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            random_state=42
        )
        
        # Train with early stopping
        try:
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return None
        
        # Make predictions
        try:
            predictions = model.predict(X_test_scaled)
            
            # Ensure predictions are non-negative
            predictions = np.maximum(predictions, 0)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions),
                'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
            }
            
            # Create evaluation plot
            eval_fig = plot_forecast_evaluation(
                y_test.values,
                predictions,
                features['date'][train_size:].values,
                'XGBoost Inventory Prediction Evaluation'
            )
            
            # Calculate feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'model': model,
                'predictions': predictions,
                'metrics': metrics,
                'evaluation_plot': eval_fig,
                'feature_importance': sorted_importance
            }
            
        except Exception as e:
            print(f"Error in prediction or evaluation: {str(e)}")
            return None
    
    except Exception as e:
        print(f"Error in XGBoost model: {str(e)}")
        return None

def train_revenue_model(data, target_col='revenue', date_col='date', product_id=None):
    """Train XGBoost model for revenue forecasting"""
    try:
        # Prepare data with normalization
        ts_data, scaler = prepare_time_series_data(data, target_col, date_col, product_id, normalize=True)
        
        if len(ts_data) < 10:
            return None
            
        # Create features
        features = pd.DataFrame()
        features['date'] = ts_data[date_col]
        features['target'] = ts_data[target_col]
        
        # Add time-based features
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        features['day_of_month'] = features['date'].dt.day
        features['quarter'] = features['date'].dt.quarter
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Create lag features
        for lag in [1, 7, 14]:
            features[f'lag_{lag}'] = features['target'].shift(lag)
            features[f'ma_{lag}'] = features['target'].rolling(window=lag).mean()
        
        # Drop rows with NaN from lag features
        features = features.dropna()
        
        # Prepare X and y
        feature_cols = ['day_of_week', 'month', 'day_of_month', 'quarter', 'is_weekend'] + \
                      [col for col in features.columns if col.startswith(('lag_', 'ma_'))]
        
        X = features[feature_cols]
        y = features['target']
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for more stable predictions
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            dates_test = features['date'].iloc[test_idx]
            
            # Scale features
            scaler_x = RobustScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            X_test_scaled = scaler_x.transform(X_test)
            
            # Train model with conservative parameters
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            fold_predictions = model.predict(X_test_scaled)
            
            all_predictions.extend(fold_predictions)
            all_actuals.extend(y_test)
            all_dates.extend(dates_test)
        
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        dates = np.array(all_dates)
        
        # Denormalize predictions and actuals
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(actuals, predictions, 'Revenue XGBoost')
        
        # Create evaluation plot
        eval_plot = plot_forecast_evaluation(
            actuals,
            predictions,
            dates,
            'Revenue Forecast Evaluation'
        )
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': metrics,
            'evaluation_plot': eval_plot,
            'feature_importance': sorted_importance
        }
    
    except Exception as e:
        print(f"Error in revenue model: {str(e)}")
        return None

def train_revenue_forecast(data, target_col='revenue', date_col='date'):
    """Train Prophet model for revenue forecasting"""
    try:
        # Prepare data for Prophet
        ts_data = prepare_time_series_data(data, target_col, date_col)
        
        # Rename columns for Prophet
        prophet_data = ts_data.rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Scale the target variable
        scaler = MinMaxScaler()
        prophet_data['y'] = scaler.fit_transform(prophet_data[['y']])
        
        # Split data
        train_size = int(len(prophet_data) * 0.8)
        train = prophet_data[:train_size]
        test = prophet_data[train_size:]
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(train)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        
        # Get predictions for test period
        predictions = forecast.tail(len(test))['yhat']
        
        # Inverse transform predictions and actual values
        predictions = scaler.inverse_transform(predictions.values.reshape(-1, 1)).flatten()
        actual = scaler.inverse_transform(test['y'].values.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(actual, predictions),
            'rmse': np.sqrt(mean_squared_error(actual, predictions)),
            'mae': mean_absolute_error(actual, predictions),
            'r2': r2_score(actual, predictions),
            'mape': np.mean(np.abs((actual - predictions) / actual)) * 100
        }
        
        return model, predictions, metrics, actual, scaler
        
    except Exception as e:
        print(f"Error in revenue forecast: {str(e)}")
        return None, None, None, None, None 