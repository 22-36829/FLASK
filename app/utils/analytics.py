import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def product_segmentation(data):
    """Perform product segmentation analysis"""
    try:
        if data.empty or len(data) < 4:  # Need at least 4 data points for 4 clusters
            return data, "<div class='alert alert-warning'>Insufficient data for segmentation analysis. Please upload more data.</div>"

        # Calculate metrics for segmentation
        metrics = data.groupby('product_id').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum',
            'stock_level': 'mean',
            'price': 'mean'
        }).reset_index()

        if len(metrics) < 4:  # Check if we have enough products
            return data, "<div class='alert alert-warning'>Need at least 4 products for segmentation analysis.</div>"

        # Scale the features
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics[['revenue', 'quantity_sold', 'stock_level', 'price']])

        # Perform K-means clustering
        n_clusters = min(4, len(metrics))  # Use minimum of 4 or number of products
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        metrics['cluster'] = kmeans.fit_predict(metrics_scaled)

        # Create visualization
        fig = px.scatter(
            metrics,
            x='revenue',
            y='quantity_sold',
            color='cluster',
            size='stock_level',
            hover_data=['product_id', 'price'],
            title='Product Segmentation Analysis'
        )

        fig.update_layout(
            xaxis_title='Total Revenue',
            yaxis_title='Total Units Sold',
            showlegend=True
        )

        return metrics, fig.to_html(full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        print(f"Error in product segmentation: {str(e)}")
        return data, f"<div class='alert alert-danger'>Error performing segmentation analysis: {str(e)}</div>"

def sales_trend_analysis(data):
    """Analyze sales trends"""
    try:
        # Calculate daily sales metrics
        daily_metrics = data.groupby('date').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum',
            'reservations': 'sum'
        }).reset_index()
        
        # Calculate moving averages
        daily_metrics['revenue_ma'] = daily_metrics['revenue'].rolling(window=7).mean()
        daily_metrics['quantity_ma'] = daily_metrics['quantity_sold'].rolling(window=7).mean()
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['revenue'],
            name='Daily Revenue',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['revenue_ma'],
            name='7-day Moving Average',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Sales Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Revenue',
            showlegend=True,
            hovermode='x unified'
        )
        
        return daily_metrics, fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        print(f"Error in sales trend analysis: {str(e)}")
        return None, "<div class='alert alert-warning'>Could not generate sales trend analysis</div>"

def inventory_optimization(data):
    """Analyze inventory optimization opportunities"""
    try:
        # Calculate inventory metrics
        inventory_metrics = data.groupby('product_id').agg({
            'stock_level': ['mean', 'std'],
            'quantity_sold': 'sum',
            'reservations': 'sum'
        }).reset_index()
        
        inventory_metrics.columns = ['product_id', 'avg_stock', 'stock_std', 'total_sold', 'total_reservations']
        
        # Calculate turnover ratio and days of inventory
        inventory_metrics['turnover_ratio'] = inventory_metrics['total_sold'] / inventory_metrics['avg_stock'].replace(0, 1)
        inventory_metrics['days_of_inventory'] = inventory_metrics['avg_stock'] / (inventory_metrics['total_sold'] / 30 + 1)
        
        # Create visualization
        fig = px.scatter(
            inventory_metrics,
            x='turnover_ratio',
            y='days_of_inventory',
            size='avg_stock',
            hover_data=['product_id'],
            title='Inventory Optimization Analysis'
        )
        
        fig.update_layout(
            xaxis_title='Turnover Ratio',
            yaxis_title='Days of Inventory',
            showlegend=True,
            hovermode='closest'
        )
        
        return inventory_metrics, fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        print(f"Error in inventory optimization: {str(e)}")
        return None, "<div class='alert alert-warning'>Could not generate inventory optimization analysis</div>"

def price_elasticity_analysis(data):
    """Analyze price elasticity"""
    try:
        # Calculate price metrics
        price_metrics = data.groupby(['product_id', 'price']).agg({
            'quantity_sold': 'sum'
        }).reset_index()
        
        # Calculate price elasticity
        price_metrics['price_lag'] = price_metrics.groupby('product_id')['price'].shift(1)
        price_metrics['quantity_lag'] = price_metrics.groupby('product_id')['quantity_sold'].shift(1)
        
        # Safe calculation of changes
        price_metrics['price_change'] = (price_metrics['price'] - price_metrics['price_lag']) / price_metrics['price_lag'].replace(0, 1)
        price_metrics['quantity_change'] = (price_metrics['quantity_sold'] - price_metrics['quantity_lag']) / price_metrics['quantity_lag'].replace(0, 1)
        
        # Safe calculation of elasticity
        price_metrics['elasticity'] = price_metrics['quantity_change'] / price_metrics['price_change'].replace(0, 1)
        
        # Create visualization
        fig = px.scatter(
            price_metrics.dropna(),
            x='price_change',
            y='quantity_change',
            hover_data=['product_id', 'elasticity'],
            title='Price Elasticity Analysis'
        )
        
        fig.update_layout(
            xaxis_title='Price Change (%)',
            yaxis_title='Quantity Change (%)',
            showlegend=True,
            hovermode='closest'
        )
        
        return price_metrics, fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        print(f"Error in price elasticity analysis: {str(e)}")
        return None, "<div class='alert alert-warning'>Could not generate price elasticity analysis</div>"

def category_performance_analysis(data):
    """Analyze category performance"""
    # Calculate category metrics
    category_metrics = data.groupby('category').agg({
        'revenue': 'sum',
        'quantity_sold': 'sum',
        'stock_level': 'mean',
        'reservations': 'sum'
    }).reset_index()
    
    # Calculate market share and growth
    total_revenue = category_metrics['revenue'].sum()
    category_metrics['market_share'] = category_metrics['revenue'] / total_revenue * 100
    
    # Create visualization
    fig = go.Figure(data=[
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['revenue'],
            name='Revenue'
        ),
        go.Bar(
            x=category_metrics['category'],
            y=category_metrics['market_share'],
            name='Market Share (%)',
            yaxis='y2'
        )
    ])
    
    fig.update_layout(
        title='Category Performance Analysis',
        yaxis=dict(title='Revenue'),
        yaxis2=dict(title='Market Share (%)', overlaying='y', side='right')
    )
    
    return category_metrics, fig 