import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_forecast_metrics(y_true, y_pred, model_name):
    """Calculate common forecasting metrics with robust error handling"""
    try:
        # Convert inputs to numpy arrays and ensure they are numeric
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        # Replace any infinite values
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Safe R² calculation
        try:
            r2 = r2_score(y_true, y_pred)
            # Clip R² to prevent extreme negative values
            r2 = max(r2, -1.0)
        except:
            r2 = -1.0  # Default value if calculation fails
        
        # Safe MAPE calculation
        try:
            # Avoid division by zero in MAPE calculation
            mask = y_true != 0
            if mask.any():
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                # Clip MAPE to reasonable range
                mape = min(mape, 1000.0)  # Cap at 1000%
            else:
                mape = 0.0
        except:
            mape = 0.0  # Default value if calculation fails
        
        return {
            'model_name': model_name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'model_name': model_name,
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'r2': -1.0,
            'mape': 0.0
        }

def plot_forecast_evaluation(y_true, y_pred, dates, title="Forecast Evaluation"):
    """Create evaluation plots for forecasts"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Actual vs Predicted',
            'Residuals Over Time',
            'Residual Distribution',
            'Prediction Error %'
        )
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=dates, y=y_true, name='Actual', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=y_pred, name='Predicted', mode='lines'),
        row=1, col=1
    )
    
    # Residuals over time
    residuals = y_true - y_pred
    fig.add_trace(
        go.Scatter(x=dates, y=residuals, mode='lines', name='Residuals'),
        row=1, col=2
    )
    
    # Residual distribution
    fig.add_trace(
        go.Histogram(x=residuals, name='Residual Distribution'),
        row=2, col=1
    )
    
    # Prediction error percentage
    error_pct = (y_true - y_pred) / y_true * 100
    fig.add_trace(
        go.Scatter(x=dates, y=error_pct, mode='lines', name='Error %'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=True
    )
    
    return fig

def create_metrics_table(metrics_dict):
    """Create an HTML table from metrics dictionary"""
    table_html = """
    <table class="table table-sm">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
    """
    
    metrics_display = {
        'mse': 'Mean Squared Error',
        'rmse': 'Root Mean Squared Error',
        'mae': 'Mean Absolute Error',
        'r2': 'R² Score',
        'mape': 'Mean Absolute Percentage Error'
    }
    
    for key, display_name in metrics_display.items():
        if key in metrics_dict:
            value = metrics_dict[key]
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            table_html += f"""
            <tr>
                <td>{display_name}</td>
                <td>{formatted_value}</td>
            </tr>
            """
    
    table_html += """
        </tbody>
    </table>
    """
    
    return table_html 