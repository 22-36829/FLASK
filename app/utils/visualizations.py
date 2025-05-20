import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import matplotlib
# Use Agg backend to avoid thread-related warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime, timedelta
import logging
from flask import send_file, make_response
import warnings
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from .model_loader import model_manager
from .evaluation_metrics import (
    calculate_revenue_metrics,
    calculate_inventory_metrics,
    calculate_reservation_metrics,
    calculate_price_metrics
)

# Set the style for matplotlib
sns.set_style("whitegrid")

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)  # Close the figure to free memory
    return img_str

def load_data():
    """Load data from CSV file"""
    try:
        # List of possible file locations in order of preference
        possible_locations = [
            os.path.join('uploads', 'latest_data.csv'),  # First try the latest uploaded file
            os.path.join('FLASK MODELS', 'pharma_forecasting_dataset.csv'),  # Then try the processed file
            'pharma_forecasting_dataset.csv',  # Then try root directory
            os.path.join('data', 'pharma_forecasting_dataset.csv'),  # Then try data directory
            os.path.join('..', 'FLASK MODELS', 'pharma_forecasting_dataset.csv')  # Finally try parent directory
        ]
        
        data_file = None
        for loc in possible_locations:
            if os.path.exists(loc):
                data_file = loc
                break
                
        if data_file is None:
            print("Could not find data file in any expected location")
            return None
        
        # Read the CSV file
        data = pd.read_csv(data_file)
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate unit_cost (assuming 60% margin on average)
        if 'unit_cost' not in data.columns and 'price' in data.columns:
            data['unit_cost'] = data['price'] * 0.4
        
        # Ensure required columns exist
        required_columns = ['date', 'product_id', 'revenue', 'quantity_sold', 'stock_level', 'reservations', 'price']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return None
        
        # Sort by date
        data = data.sort_values('date')
        
        print(f"Successfully loaded {len(data)} rows of data from {data_file}")
        return data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def filter_data(data, date_range=None, category=None, product_id=None):
    """Filter data based on selected criteria"""
    try:
        if data is None or data.empty:
            return pd.DataFrame()
            
        filtered_data = data.copy()
        
        # Apply date range filter
        if date_range:
            if isinstance(date_range, str):
                # Convert dates to timezone-naive datetime for consistent comparison
                data_end_date = pd.to_datetime(filtered_data['date'].max()).replace(hour=23, minute=59, second=59)
                data_start_date = pd.to_datetime(filtered_data['date'].min())
                
                # Handle predefined ranges
                if date_range == 'Last 7 days':
                    start_date = data_end_date - pd.Timedelta(days=6)  # -6 because we want 7 days including today
                    start_date = start_date.replace(hour=0, minute=0, second=0)
                    end_date = data_end_date
                elif date_range == 'Last 30 days':
                    start_date = data_end_date - pd.Timedelta(days=29)  # -29 because we want 30 days including today
                    start_date = start_date.replace(hour=0, minute=0, second=0)
                    end_date = data_end_date
                elif date_range == 'Last 90 days':
                    start_date = data_end_date - pd.Timedelta(days=89)  # -89 because we want 90 days including today
                    start_date = start_date.replace(hour=0, minute=0, second=0)
                    end_date = data_end_date
                elif date_range == 'Last 365 days':
                    start_date = data_end_date - pd.Timedelta(days=364)  # -364 because we want 365 days including today
                    start_date = start_date.replace(hour=0, minute=0, second=0)
                    end_date = data_end_date
                elif date_range.startswith('custom:'):
                    # Handle custom date range in format 'custom:YYYY-MM-DD,YYYY-MM-DD'
                    try:
                        start_str, end_str = date_range.replace('custom:', '').split(',')
                        start_date = pd.to_datetime(start_str).replace(hour=0, minute=0, second=0)
                        end_date = pd.to_datetime(end_str).replace(hour=23, minute=59, second=59)
                        
                        # Ensure dates are within data range
                        start_date = max(start_date, data_start_date)
                        end_date = min(end_date, data_end_date)
                    except Exception as e:
                        print(f"Error parsing custom date range: {str(e)}")
                        # Default to last 30 days if custom range is invalid
                        start_date = data_end_date - pd.Timedelta(days=29)
                        start_date = start_date.replace(hour=0, minute=0, second=0)
                        end_date = data_end_date
                else:
                    # Default to last 30 days if invalid range
                    start_date = data_end_date - pd.Timedelta(days=29)
                    start_date = start_date.replace(hour=0, minute=0, second=0)
                    end_date = data_end_date
            else:
                # Assume date_range is a tuple of (start_date, end_date)
                start_date = pd.to_datetime(date_range[0]).replace(hour=0, minute=0, second=0)
                end_date = pd.to_datetime(date_range[1]).replace(hour=23, minute=59, second=59)
            
            # Convert filtered_data['date'] to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(filtered_data['date']):
                filtered_data['date'] = pd.to_datetime(filtered_data['date'])
            
            # Apply date filter
            filtered_data = filtered_data[
                (filtered_data['date'] >= start_date) & 
                (filtered_data['date'] <= end_date)
            ]
            
            # Sort by date
            filtered_data = filtered_data.sort_values('date')
        
        # Apply category filter
        if category and category != 'All Categories' and 'category' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['category'] == category]
        
        # Apply product filter
        if product_id and product_id != '':
            filtered_data = filtered_data[filtered_data['product_id'] == product_id]
        
        return filtered_data
        
    except Exception as e:
        print(f"Error filtering data: {str(e)}")
        return pd.DataFrame()

def total_sales_over_time(data):
    """Generate total sales over time visualization"""
    sales_over_time = data.groupby('date')['revenue'].sum().reset_index()
    
    fig = px.line(
        sales_over_time, 
        x='date', 
        y='revenue',
        title='Total Sales Over Time',
        labels={'date': 'Date', 'revenue': 'Revenue ($)'}
    )
    
    return fig

def inventory_turnover_per_product(data):
    """Create inventory turnover chart"""
    try:
        # Calculate inventory turnover metrics
        turnover_metrics = data.groupby(['date', 'product_id']).agg({
            'quantity_sold': 'sum',
            'stock_level': 'mean'
        }).reset_index()
        
        turnover_metrics['turnover_ratio'] = turnover_metrics['quantity_sold'] / turnover_metrics['stock_level'].replace(0, 1)
        
        # Create visualization
        fig = go.Figure()
        
        for product_id in turnover_metrics['product_id'].unique():
            product_data = turnover_metrics[turnover_metrics['product_id'] == product_id]
            
            fig.add_trace(go.Scatter(
                x=product_data['date'],
                y=product_data['turnover_ratio'],
                name=f'Product {product_id}',
                mode='lines'
            ))
        
        fig.update_layout(
            title='Inventory Turnover by Product',
            xaxis_title='Date',
            yaxis_title='Turnover Ratio',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating inventory turnover chart: {str(e)}</div>"

def top_10_reserved_products(data):
    """Generate top 10 reserved products visualization"""
    top_reservations = data.groupby('product_id').agg({
        'reservations': 'sum',
        'product_name': 'first'
    }).sort_values('reservations', ascending=False).head(10).reset_index()
    
    fig = px.bar(
        top_reservations,
        x='product_name',
        y='reservations',
        title='Top 10 Reserved Products',
        labels={'product_name': 'Product', 'reservations': 'Total Reservations'}
    )
    
    return fig

def price_fluctuation_trends(data):
    """Create price fluctuation trend chart"""
    try:
        if 'price' not in data.columns:
            raise ValueError("Price column not found in data")
            
        # Calculate daily average prices
        daily_prices = data.groupby('date')['price'].agg(['mean', 'min', 'max']).reset_index()
        
        # Create visualization
        fig = go.Figure()
        
        # Add mean price line
        fig.add_trace(go.Scatter(
            x=daily_prices['date'],
            y=daily_prices['mean'],
            name='Average Price',
            line=dict(color='blue')
        ))
        
        # Add price range
        fig.add_trace(go.Scatter(
            x=daily_prices['date'],
            y=daily_prices['max'],
            name='Price Range',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_prices['date'],
            y=daily_prices['min'],
            name='Price Range',
            fill='tonexty',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.update_layout(
            title='Price Fluctuation Trends',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"Error generating price trend chart: {str(e)}")
        return f"<div class='alert alert-warning'>Error generating price trend chart: {str(e)}</div>"

def sales_per_employee(data):
    """Create sales per employee chart"""
    try:
        # Calculate sales per employee
        if 'employee_id' not in data.columns:
            # Create a mock segmentation if employee data is not available
            data['segment'] = pd.qcut(data['revenue'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
            
            segment_metrics = data.groupby('segment').agg({
                'revenue': 'sum',
                'quantity_sold': 'sum'
            }).reset_index()
            
            # Create a treemap
            fig = go.Figure(go.Treemap(
                labels=segment_metrics['segment'],
                parents=[''] * len(segment_metrics),
                values=segment_metrics['revenue'],
                textinfo='label+value',
                hovertemplate='Segment: %{label}<br>Revenue: $%{value:,.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Market Segmentation by Revenue',
                width=600,
                height=450
            )
        else:
            # Use actual employee data if available
            employee_metrics = data.groupby('employee_id').agg({
                'revenue': 'sum',
                'quantity_sold': 'sum'
            }).reset_index()
            
            fig = go.Figure(go.Bar(
                x=employee_metrics['employee_id'],
                y=employee_metrics['revenue'],
                text=employee_metrics['revenue'].apply(lambda x: f'${x:,.2f}'),
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Sales Performance by Employee',
                xaxis_title='Employee ID',
                yaxis_title='Total Revenue ($)',
                showlegend=False,
                hovermode='x unified'
            )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating segmentation chart: {str(e)}</div>"

def generate_all_visualizations(data):
    """Generate all visualizations for the dashboard"""
    if data.empty:
        return {
            'revenue': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>',
            'inventory': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>',
            'reservations': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>',
            'comparison': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>'
        }

    try:
        # Revenue Chart
        revenue_fig = go.Figure()
        daily_revenue = data.groupby('date')['revenue'].sum().reset_index()
        revenue_fig.add_trace(
            go.Scatter(
                x=daily_revenue['date'],
                y=daily_revenue['revenue'],
                mode='lines+markers',
                name='Daily Revenue',
                hovertemplate='Date: %{x}<br>Revenue: $%{y:.2f}<extra></extra>'
            )
        )
        revenue_fig.update_layout(
            title='Daily Revenue',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )

        # Inventory Chart
        inventory_fig = go.Figure()
        product_inventory = data.groupby('product_id').agg({
            'stock_level': 'mean',
            'product_name': 'first'
        }).reset_index()
        inventory_fig.add_trace(
            go.Bar(
                x=product_inventory['product_name'],
                y=product_inventory['stock_level'],
                name='Average Stock Level',
                hovertemplate='Product: %{x}<br>Stock Level: %{y:.0f}<extra></extra>'
            )
        )
        inventory_fig.update_layout(
            title='Current Inventory Levels',
            xaxis_title='Product',
            yaxis_title='Stock Level',
            hovermode='x unified'
        )

        # Reservations Chart
        reservations_fig = go.Figure()
        daily_reservations = data.groupby('date')['reservations'].sum().reset_index()
        reservations_fig.add_trace(
            go.Scatter(
                x=daily_reservations['date'],
                y=daily_reservations['reservations'],
                mode='lines+markers',
                name='Daily Reservations',
                hovertemplate='Date: %{x}<br>Reservations: %{y:.0f}<extra></extra>'
            )
        )
        reservations_fig.update_layout(
            title='Daily Reservations',
            xaxis_title='Date',
            yaxis_title='Number of Reservations',
            hovermode='x unified'
        )

        # Product Comparison Chart
        comparison_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Revenue by Product',
                'Sales Volume by Product',
                'Average Stock Level',
                'Total Reservations'
            )
        )

        product_metrics = data.groupby('product_id').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum',
            'stock_level': 'mean',
            'reservations': 'sum',
            'product_name': 'first'
        }).reset_index()

        # Revenue subplot
        comparison_fig.add_trace(
            go.Bar(
                x=product_metrics['product_name'],
                y=product_metrics['revenue'],
                name='Revenue',
                hovertemplate='Product: %{x}<br>Revenue: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Sales volume subplot
        comparison_fig.add_trace(
            go.Bar(
                x=product_metrics['product_name'],
                y=product_metrics['quantity_sold'],
                name='Units Sold',
                hovertemplate='Product: %{x}<br>Units Sold: %{y:.0f}<extra></extra>'
            ),
            row=1, col=2
        )

        # Stock level subplot
        comparison_fig.add_trace(
            go.Bar(
                x=product_metrics['product_name'],
                y=product_metrics['stock_level'],
                name='Stock Level',
                hovertemplate='Product: %{x}<br>Stock Level: %{y:.0f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Reservations subplot
        comparison_fig.add_trace(
            go.Bar(
                x=product_metrics['product_name'],
                y=product_metrics['reservations'],
                name='Reservations',
                hovertemplate='Product: %{x}<br>Reservations: %{y:.0f}<extra></extra>'
            ),
            row=2, col=2
        )

        comparison_fig.update_layout(
            height=800,
            showlegend=False,
            title_text='Product Performance Comparison'
        )

        return {
            'revenue': revenue_fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'inventory': inventory_fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'reservations': reservations_fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'comparison': comparison_fig.to_html(full_html=False, include_plotlyjs='cdn')
        }

    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        return {
            'revenue': f'<div class="alert alert-danger">Error generating revenue chart: {str(e)}</div>',
            'inventory': f'<div class="alert alert-danger">Error generating inventory chart: {str(e)}</div>',
            'reservations': f'<div class="alert alert-danger">Error generating reservations chart: {str(e)}</div>',
            'comparison': f'<div class="alert alert-danger">Error generating comparison chart: {str(e)}</div>'
        }

def generate_forecast_visualizations(product_id=None, date_range='Last 30 days', category='All Categories'):
    """Generate forecast visualizations for the dashboard"""
    # For now, return placeholder HTML
    revenue_forecast = "<div class='alert alert-info'>Revenue forecast will appear here</div>"
    inventory_forecast = "<div class='alert alert-info'>Inventory forecast will appear here</div>"
    reservations_forecast = "<div class='alert alert-info'>Reservations forecast will appear here</div>"
    
    return {
        'revenue': revenue_forecast,
        'inventory': inventory_forecast,
        'reservations': reservations_forecast
    }

def get_product_list(data=None):
    """Get list of products for dropdown filter"""
    if data is None:
        data = load_data()
    
    products = [{'id': row['product_id'], 'name': row['product_name']} 
                for _, row in data.drop_duplicates('product_id').iterrows()]
    
    return products

def export_chart_to_html(fig, filename):
    """Export chart to HTML file"""
    html_path = os.path.join('static', 'exports', filename)
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    
    fig.write_html(html_path)
    
    return send_file(html_path, as_attachment=True)

def export_data_to_csv(data, filename):
    """Export data to CSV file"""
    csv_path = os.path.join('static', 'exports', filename)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    data.to_csv(csv_path, index=False)
    
    return send_file(csv_path, as_attachment=True)

def export_chart_to_image(fig, filename, format='png'):
    """Export chart to image file"""
    img_path = os.path.join('static', 'exports', filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    fig.write_image(img_path, format=format)
    
    return send_file(img_path, as_attachment=True)

def create_interactive_revenue_chart(data, product_id=None):
    """Create interactive revenue chart"""
    try:
        if product_id:
            data = data[data['product_id'] == product_id]
        
        # Calculate metrics using imported function
        metrics = calculate_revenue_metrics(data)
        
        # Calculate daily revenue
        daily_revenue = data.groupby('date')['revenue'].sum().reset_index()
        
        # Calculate moving average
        daily_revenue['MA7'] = daily_revenue['revenue'].rolling(window=7).mean()
        daily_revenue['MA30'] = daily_revenue['revenue'].rolling(window=30).mean()
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['revenue'],
            name='Daily Revenue',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['MA7'],
            name='7-day Moving Average',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['MA30'],
            name='30-day Moving Average',
            line=dict(color='green', dash='dot')
        ))
        
        fig.update_layout(
            title='Revenue Over Time',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn'), metrics
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating revenue chart: {str(e)}</div>", {'mape': 'N/A', 'r2': 'N/A'}

def create_interactive_inventory_chart(data, product_id=None):
    """Create interactive inventory chart"""
    try:
        if product_id:
            data = data[data['product_id'] == product_id]
        
        # Calculate metrics using imported function
        metrics = calculate_inventory_metrics(data)
        
        # Calculate daily inventory metrics
        daily_inventory = data.groupby('date').agg({
            'stock_level': 'mean',
            'quantity_sold': 'sum',
            'reservations': 'sum'
        }).reset_index()
        
        # Calculate inventory turnover
        daily_inventory['turnover_ratio'] = daily_inventory['quantity_sold'] / daily_inventory['stock_level'].replace(0, 1)
        
        # Create visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=daily_inventory['date'],
                y=daily_inventory['stock_level'],
                name='Stock Level',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_inventory['date'],
                y=daily_inventory['turnover_ratio'],
                name='Turnover Ratio',
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Inventory Status Over Time',
            xaxis_title='Date',
            yaxis_title='Stock Level',
            yaxis2_title='Turnover Ratio',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn'), metrics
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating inventory chart: {str(e)}</div>", {'mae': 'N/A', 'accuracy': 'N/A'}

def create_interactive_reservations_chart(data, product_id=None):
    """Create interactive reservations chart"""
    try:
        if product_id:
            data = data[data['product_id'] == product_id]
        
        # Calculate metrics using imported function
        metrics = calculate_reservation_metrics(data)
        
        # Calculate daily reservations
        daily_reservations = data.groupby('date')['reservations'].sum().reset_index()
        
        # Calculate moving average
        daily_reservations['MA7'] = daily_reservations['reservations'].rolling(window=7).mean()
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_reservations['date'],
            y=daily_reservations['reservations'],
            name='Daily Reservations',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_reservations['date'],
            y=daily_reservations['MA7'],
            name='7-day Moving Average',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Reservations Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Reservations',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn'), metrics
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating reservations chart: {str(e)}</div>", {'rmse': 'N/A', 'precision': 'N/A'}

def create_product_comparison_chart(data):
    """Create interactive product comparison chart"""
    try:
        # Calculate product metrics
        product_metrics = data.groupby('product_id').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum',
            'stock_level': 'mean',
            'reservations': 'sum',
            'product_name': 'first'
        }).reset_index()
        
        # Calculate derived metrics
        product_metrics['turnover_ratio'] = product_metrics['quantity_sold'] / product_metrics['stock_level'].replace(0, 1)
        product_metrics['revenue_per_unit'] = product_metrics['revenue'] / product_metrics['quantity_sold'].replace(0, 1)
        
        # Create visualization
        fig = px.scatter(
            product_metrics,
            x='revenue',
            y='quantity_sold',
            size='stock_level',
            color='turnover_ratio',
            hover_data=['product_name', 'reservations', 'revenue_per_unit'],
            title='Product Performance Comparison'
        )
        
        fig.update_layout(
            xaxis_title='Total Revenue ($)',
            yaxis_title='Total Units Sold',
            showlegend=True,
            hovermode='closest'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating product comparison chart: {str(e)}</div>"

def sales_trend_analysis(data):
    """Generate sales trend analysis visualization"""
    try:
        # Calculate daily sales metrics
        daily_metrics = data.groupby('date').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum'
        }).reset_index()
        
        # Calculate moving averages
        daily_metrics['revenue_ma7'] = daily_metrics['revenue'].rolling(window=7).mean()
        daily_metrics['revenue_ma30'] = daily_metrics['revenue'].rolling(window=30).mean()
        daily_metrics['quantity_ma7'] = daily_metrics['quantity_sold'].rolling(window=7).mean()
        
        # Create visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue traces
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['date'],
                y=daily_metrics['revenue'],
                name='Daily Revenue',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['date'],
                y=daily_metrics['revenue_ma7'],
                name='7-day Revenue MA',
                line=dict(color='red', dash='dash')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['date'],
                y=daily_metrics['revenue_ma30'],
                name='30-day Revenue MA',
                line=dict(color='green', dash='dot')
            ),
            secondary_y=False
        )
        
        # Add quantity sold trace
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['date'],
                y=daily_metrics['quantity_ma7'],
                name='7-day Units Sold MA',
                line=dict(color='orange', dash='dash')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Sales Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            yaxis2_title='Units Sold',
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        return f"<div class='alert alert-warning'>Error generating sales trend chart: {str(e)}</div>" 