import pandas as pd
import numpy as np
from .improved_models import ImprovedForecaster
import logging
import joblib
import os

logger = logging.getLogger(__name__)

def train_all_models(data_path):
    """
    Train all models for each product in the dataset
    """
    try:
        # Read data
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        
        # Create forecaster instance
        forecaster = ImprovedForecaster()
        
        # Get unique products
        products = data['product_id'].unique()
        
        results = {
            'sales': {},
            'inventory': {},
            'revenue': {}
        }
        
        for product_id in products:
            logger.info(f"Training models for product {product_id}")
            
            # Train sales model
            sales_success = forecaster.train_product_sales_model(data, product_id)
            if sales_success:
                results['sales'][product_id] = forecaster.metrics[f'sales_{product_id}']
            
            # Train inventory model
            inv_success = forecaster.train_inventory_model(data, product_id)
            if inv_success:
                results['inventory'][product_id] = forecaster.metrics[f'inventory_{product_id}']
            
            # Train revenue model
            rev_success = forecaster.train_revenue_model(data, product_id)
            if rev_success:
                results['revenue'][product_id] = forecaster.metrics[f'revenue_{product_id}']
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(forecaster, 'models/forecaster.joblib')
        
        # Print results
        print("\nTraining Results:")
        print("================")
        
        for forecast_type, metrics in results.items():
            print(f"\n{forecast_type.title()} Models:")
            print("-" * 50)
            
            accuracies = [m['accuracy'] for m in metrics.values()]
            avg_accuracy = np.mean(accuracies)
            
            print(f"Average Accuracy: {avg_accuracy:.2f}%")
            print("\nPer Product Metrics:")
            
            for product_id, metric in metrics.items():
                print(f"\n{product_id}:")
                print(f"  Accuracy: {metric['accuracy']:.2f}%")
                print(f"  MAE: {metric['mae']:.2f}")
                print(f"  RMSE: {metric['rmse']:.2f}")
                print(f"  R² Score: {metric['r2']:.2f}")
        
        return forecaster, results
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train models
    forecaster, results = train_all_models('pharma_forecasting_dataset.csv') 