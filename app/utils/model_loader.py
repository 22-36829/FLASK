import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import pickle

class ModelManager:
    """Class to manage loading and using trained models"""
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      "FLASK MODELS", "saved_models")
        self.revenue_models = {}
        self.inventory_models = {}
        self.reservation_models = {}
    
    def load_models(self):
        """Load all available models from the saved_models directory"""
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
                print(f"Created models directory: {self.models_dir}")
                return
            
            # Load models from saved_models directory
            for file in os.listdir(self.models_dir):
                if not file.endswith('.pkl'):
                    continue
                
                model_path = os.path.join(self.models_dir, file)
                
                try:
                    # Try different loading methods
                    model = None
                    errors = []
                    
                    # Method 1: Try joblib with custom numpy path mapping
                    try:
                        with open(model_path, 'rb') as f:
                            # Create a custom unpickler to handle numpy version differences
                            class CustomUnpickler(pickle.Unpickler):
                                def find_class(self, module, name):
                                    # Remap old numpy paths to new ones
                                    if module.startswith('numpy.'):
                                        try:
                                            module = module.replace('numpy._core', 'numpy.core')
                                            module = module.replace('numpy.core.numeric', 'numpy')
                                            module = module.replace('numpy.core.multiarray', 'numpy')
                                            return super().find_class(module, name)
                                        except:
                                            return super().find_class('numpy', name)
                                    return super().find_class(module, name)
                            
                            model = CustomUnpickler(f).load()
                    except Exception as e:
                        errors.append(f"custom unpickler error: {str(e)}")
                    
                    # Method 2: Try direct pickle load
                    if model is None:
                        try:
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                        except Exception as e:
                            errors.append(f"pickle error: {str(e)}")
                    
                    # Method 3: For XGBoost models, try loading with xgb.Booster
                    if model is None and 'xgboost' in file:
                        try:
                            import xgboost as xgb
                            model = xgb.Booster()
                            model.load_model(model_path)
                        except Exception as e:
                            errors.append(f"xgboost error: {str(e)}")
                    
                    if model is None:
                        print(f"Error loading model {file}: Failed to load model with all methods: {'; '.join(errors)}")
                        continue
                    
                    # Store the model in the appropriate dictionary
                    if 'arima_revenue' in file:
                        product_id = file.replace('arima_revenue_', '').replace('.pkl', '')
                        self.revenue_models[product_id] = model
                        print(f"Loaded ARIMA revenue model for {product_id}")
                    elif 'xgboost_inventory' in file:
                        self.inventory_models['all'] = model
                        print("Loaded XGBoost inventory model")
                    elif 'prophet_reservations' in file:
                        product_id = file.replace('prophet_reservations_', '').replace('.pkl', '')
                        self.reservation_models[product_id] = model
                        print(f"Loaded Prophet reservation model for {product_id}")
                    
                except Exception as e:
                    print(f"Error loading model {file}: {str(e)}")
                    continue
            
            # Create fallback aggregate models if needed
            if not self.revenue_models.get('all_products'):
                print("Created fallback aggregate ARIMA model")
                self.revenue_models['all_products'] = None
            
            if not self.inventory_models.get('all'):
                print("Created fallback aggregate XGBoost model")
                self.inventory_models['all'] = None
            
            if not self.reservation_models.get('all_products'):
                print("Created fallback aggregate Prophet model")
                self.reservation_models['all_products'] = None
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return
    
    def predict_revenue(self, product_id, forecast_days=30):
        """Generate revenue forecast using ARIMA model"""
        try:
            # Check if we have a trained model for this product
            if product_id not in self.revenue_models:
                print(f"No trained ARIMA model found for product {product_id}")
                return None
            
            # Use the trained model
            model = self.revenue_models[product_id]
            forecast = model.forecast(steps=forecast_days)
            
            # Create forecast dataframe
            dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_df = pd.DataFrame({
                'date': dates,
                'forecasted_revenue': forecast,
                'lower_bound': forecast * 0.9,  # 90% confidence interval
                'upper_bound': forecast * 1.1   # 110% confidence interval
            })
            
            return forecast_df
        except Exception as e:
            print(f"Error predicting revenue for product {product_id}: {str(e)}")
            return None
    
    def predict_inventory(self, features):
        """Generate inventory depletion forecast using XGBoost model"""
        try:
            if 'all' not in self.inventory_models:
                print("No trained XGBoost inventory model found")
                return None
            
            model = self.inventory_models['all']
            
            # Prepare features for prediction
            feature_cols = model.get_booster().feature_names
            if not all(col in features.columns for col in feature_cols):
                print("Missing required features for inventory prediction")
                return None
            
            X = features[feature_cols]
            predictions = model.predict(X)
            
            return predictions
        except Exception as e:
            print(f"Error predicting inventory: {str(e)}")
            return None
    
    def predict_reservations(self, product_id, forecast_days=30):
        """Generate reservations forecast using Prophet model"""
        try:
            # Check if we have a trained model for this product
            if product_id not in self.reservation_models:
                print(f"No trained Prophet model found for product {product_id}")
                return None
            
            # Use the trained model
            model = self.reservation_models[product_id]
            
            # Create future dataframe for Prophet
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Filter only future dates
            forecast = forecast.iloc[-forecast_days:]
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast['ds'],
                'forecasted_reservations': forecast['yhat'],
                'lower_bound': forecast['yhat_lower'],
                'upper_bound': forecast['yhat_upper']
            })
            
            return forecast_df
        except Exception as e:
            print(f"Error predicting reservations for product {product_id}: {str(e)}")
            return None

# Create a singleton instance
model_manager = ModelManager()
model_manager.load_models() 