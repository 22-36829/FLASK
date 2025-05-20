import os
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_from_directory
from flask_wtf.csrf import CSRFProtect
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import logging
from .utils.filters import format_number
from .utils.visualizations import (
    load_data, filter_data, total_sales_over_time, inventory_turnover_per_product,
    top_10_reserved_products, price_fluctuation_trends, sales_per_employee,
    generate_all_visualizations, get_product_list, export_chart_to_html,
    export_data_to_csv, export_chart_to_image,
    create_interactive_revenue_chart,
    create_interactive_inventory_chart,
    create_interactive_reservations_chart,
    create_product_comparison_chart,
    generate_forecast_visualizations,
    sales_trend_analysis
)
from .utils.model_processor import (
    process_uploaded_file, generate_arima_sales_forecast, generate_inventory_forecast,
    generate_reservations_forecast, generate_model_metrics
)
from .utils.analytics import (
    product_segmentation, inventory_optimization,
    price_elasticity_analysis, category_performance_analysis
)
from .utils.evaluation_metrics import (
    calculate_sales_metrics, calculate_price_metrics,
    calculate_revenue_metrics, calculate_inventory_metrics,
    calculate_reservation_metrics
)
from .utils.model_loader import model_manager
from .utils.model_evaluation import create_metrics_table
from .utils.forecasting import (
    train_arima_model, train_xgboost_model, train_revenue_forecast
)
from .models import db, bcrypt, UploadedFile, ProductForecast, InventoryPrediction, ModelMetric
from flask_login import LoginManager, login_required, current_user
from sqlalchemy import text

migrate = Migrate()
socketio = SocketIO()
login_manager = LoginManager()

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(app.instance_path, 'flask_dashboard.sqlite'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.join(os.path.dirname(app.instance_path), 'uploads'),
        MODELS_FOLDER=os.path.join(os.path.dirname(app.instance_path), 'FLASK MODELS', 'saved_models'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    bcrypt.init_app(app)
    csrf = CSRFProtect(app)
    migrate.init_app(app, db)
    socketio.init_app(app)
    
    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(id):
        from .models import User
        return User.query.get(int(id))

    # Register filters
    app.jinja_env.filters['format_number'] = format_number

    # Add context processor for current year
    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.now().year}

    # Register blueprints
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # Register routes
    @app.route('/')
    @login_required
    def dashboard():
        try:
            # Get filter parameters
            date_range = request.args.get('date_range', 'Last 7 days')
            category = request.args.get('category', 'All Categories')
            product_id = request.args.get('product_id', '')
            custom_date_range = request.args.get('custom_date_range', '')
            
            # Load data
            data = load_data()
            
            # Initialize empty charts dictionary
            charts = {
                'revenue': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>',
                'inventory': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>',
                'reservations': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>',
                'comparison': '<div class="alert alert-info">No data available. Please upload a CSV file.</div>'
            }
            
            # Get categories from data
            categories = ['All Categories']
            if not data.empty and 'category' in data.columns:
                categories.extend(sorted(data['category'].unique().tolist()))
            
            # Handle custom date range
            if date_range == 'custom' and custom_date_range:
                try:
                    start_str, end_str = custom_date_range.split(' to ')
                    date_range = f"custom:{start_str},{end_str}"
                except:
                    date_range = 'Last 30 days'  # Default if custom range is invalid
            
            # Only generate visualizations if we have data
            if not data.empty:
                filtered_data = filter_data(data, date_range, category, product_id)
                charts = generate_all_visualizations(filtered_data)
            
            # Get list of products for filter dropdown
            products = get_product_list(data)
            
            app.logger.info('Dashboard loaded successfully')
            return render_template('dashboard.html', 
                                charts=charts,
                                products=products,
                                categories=categories,
                                selected_date_range=date_range,
                                selected_category=category,
                                selected_product=product_id,
                                custom_date_range=custom_date_range)
        except Exception as e:
            app.logger.error(f'Error loading dashboard: {str(e)}')
            flash('Error loading dashboard data', 'danger')
            return render_template('dashboard.html', 
                                error=str(e),
                                charts={
                                    'revenue': '<div class="alert alert-danger">Error loading chart.</div>',
                                    'inventory': '<div class="alert alert-danger">Error loading chart.</div>',
                                    'reservations': '<div class="alert alert-danger">Error loading chart.</div>',
                                    'comparison': '<div class="alert alert-danger">Error loading chart.</div>'
                                })

    @app.route('/pharma_dashboard')
    @login_required
    def pharma_dashboard():
        # Get filter parameters
        source_file_id = request.args.get('source_file_id', None)
        selected_product = request.args.get('product_id', '')
        date_range = request.args.get('date_range', 'Last 30 days')
        category = request.args.get('category', 'All Categories')
        
        # Generate dummy data for now
        data = load_data()
        products = get_product_list(data)
        
        return render_template('pharma_dashboard.html',
                              products=products,
                              selected_product=selected_product,
                              selected_date_range=date_range,
                              selected_category=category)

    @app.route('/analytics')
    @login_required
    def analytics():
        try:
            # Get filter parameters
            date_range = request.args.get('date_range', 'Last 30 days')
            category = request.args.get('category', 'All Categories')
            product_id = request.args.get('product_id', '')
            
            # Load and filter data
            data = load_data()
            if data is None or data.empty:
                return render_template('analytics.html',
                                    error="No data available. Please ensure the data file exists and is properly formatted.",
                                    products=[],
                                    categories=[],
                                    selected_date_range=date_range,
                                    selected_category=category,
                                    selected_product=product_id)
            
            # Verify required columns
            required_columns = ['date', 'product_id', 'revenue', 'quantity_sold', 'stock_level', 'reservations', 'price']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return render_template('analytics.html',
                                    error=f"Missing required columns: {', '.join(missing_columns)}. Please ensure your data file contains all required columns.",
                                    products=[],
                                    categories=[],
                                    selected_date_range=date_range,
                                    selected_category=category,
                                    selected_product=product_id)
            
            filtered_data = filter_data(data, date_range, category, product_id)
            
            if filtered_data.empty:
                return render_template('analytics.html',
                                    error="No data available for the selected filters",
                                    products=get_product_list(data),
                                    categories=data['category'].unique().tolist() if not data.empty else [],
                                    selected_date_range=date_range,
                                    selected_category=category,
                                    selected_product=product_id)
            
            try:
                # Calculate unit cost for margin calculations (40% of price)
                filtered_data['unit_cost'] = filtered_data['price'] * 0.4
                
                # Generate time series visualizations with metrics
                revenue_chart, revenue_metrics = create_interactive_revenue_chart(filtered_data, product_id)
                inventory_chart, inventory_metrics = create_interactive_inventory_chart(filtered_data, product_id)
                reservations_chart, reservation_metrics = create_interactive_reservations_chart(filtered_data, product_id)
                product_comparison = create_product_comparison_chart(filtered_data)
                
                # Generate trend analysis
                sales_trend_chart = sales_trend_analysis(filtered_data)
                inventory_movement_chart = inventory_turnover_per_product(filtered_data)
                price_trend_chart = price_fluctuation_trends(filtered_data)
                segmentation_chart = sales_per_employee(filtered_data)
                
                # Calculate all metrics
                sales_metrics = calculate_sales_metrics(filtered_data)
                price_metrics = calculate_price_metrics(filtered_data)
                revenue_metrics = calculate_revenue_metrics(filtered_data)
                inventory_metrics = calculate_inventory_metrics(filtered_data)
                reservation_metrics = calculate_reservation_metrics(filtered_data)
                
            except Exception as e:
                app.logger.error(f"Error generating visualizations: {str(e)}")
                return render_template('analytics.html',
                                    error=f"Error generating visualizations: {str(e)}",
                                    products=get_product_list(data),
                                    categories=data['category'].unique().tolist(),
                                    selected_date_range=date_range,
                                    selected_category=category,
                                    selected_product=product_id)
            
            return render_template('analytics.html',
                                revenue_forecast_chart=revenue_chart,
                                inventory_forecast_chart=inventory_chart,
                                reservations_forecast_chart=reservations_chart,
                                product_comparison_chart=product_comparison,
                                sales_trend_chart=sales_trend_chart,
                                inventory_movement_chart=inventory_movement_chart,
                                price_trend_chart=price_trend_chart,
                                segmentation_chart=segmentation_chart,
                                revenue_metrics=revenue_metrics,
                                inventory_metrics=inventory_metrics,
                                reservation_metrics=reservation_metrics,
                                sales_metrics=sales_metrics,
                                price_metrics=price_metrics,
                                products=get_product_list(data),
                                categories=data['category'].unique().tolist(),
                                selected_date_range=date_range,
                                selected_category=category,
                                selected_product=product_id)
            
        except Exception as e:
            app.logger.error(f"Error in analytics: {str(e)}")
            return render_template('analytics.html',
                                error=f"An error occurred: {str(e)}",
                                products=[],
                                categories=[],
                                selected_date_range=date_range,
                                selected_category=category,
                                selected_product=product_id)

    @app.route('/upload-history')
    @login_required
    def upload_history():
        try:
            uploads = UploadedFile.query.order_by(UploadedFile.upload_date.desc()).all()
        except:
            uploads = []
        return render_template('upload_history.html', uploads=uploads)

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/upload-data', methods=['POST'])
    @login_required
    def upload_data():
        if 'data_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.referrer or url_for('dashboard'))
        
        file = request.files['data_file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.referrer or url_for('dashboard'))
        
        if not file.filename.endswith('.csv'):
            flash('Invalid file format. Please upload a CSV file.', 'danger')
            return redirect(request.referrer or url_for('dashboard'))
        
        try:
            # Save the uploaded file with a unique name to prevent overwriting
            filename = secure_filename(file.filename)
            base_name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{base_name}_{timestamp}{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Save to FLASK MODELS directory for processing
            models_dir = os.path.join('FLASK MODELS')
            os.makedirs(models_dir, exist_ok=True)
            models_path = os.path.join(models_dir, 'pharma_forecasting_dataset.csv')
            import shutil
            shutil.copy2(file_path, models_path)
            
            # Create database entry
            uploaded_file = UploadedFile(
                filename=unique_filename,
                upload_date=datetime.now(),
                status='processing'
            )
            db.session.add(uploaded_file)
            db.session.commit()
            
            try:
                # Clear any cached data
                if hasattr(app, 'cached_data'):
                    delattr(app, 'cached_data')
                
                # Process the file and generate forecasts
                from .utils.model_processor import process_uploaded_file
                forecasts = process_uploaded_file(file_path)
                
                if forecasts:
                    uploaded_file.status = 'completed'
                    flash('File successfully uploaded and processed. New forecasts have been generated.', 'success')
                else:
                    uploaded_file.status = 'error'
                    uploaded_file.error_message = 'Failed to generate forecasts'
                    flash('File uploaded but failed to generate forecasts. Please check the file format.', 'warning')
                
                db.session.commit()
                
            except Exception as e:
                app.logger.error(f'Error processing file: {str(e)}')
                uploaded_file.status = 'error'
                uploaded_file.error_message = str(e)
                db.session.commit()
                flash(f'Error processing file: {str(e)}', 'danger')
                
                # Clean up files on error
                try:
                    os.remove(file_path)
                    if os.path.exists(models_path):
                        os.remove(models_path)
                except Exception as e:
                    app.logger.error(f'Error cleaning up files: {str(e)}')
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            app.logger.error(f'Error uploading file: {str(e)}')
            flash(f'Error uploading file: {str(e)}', 'danger')
            return redirect(request.referrer or url_for('dashboard'))

    @app.route('/export_data')
    @login_required
    def export_data():
        # Get filter parameters
        date_range = request.args.get('date_range', 'Last 30 days')
        category = request.args.get('category', 'All Categories')
        product_id = request.args.get('product_id', '')
        
        # Load and filter data
        data = load_data()
        filtered_data = filter_data(data, date_range, category, product_id)
        
        # Export to CSV
        return export_data_to_csv(filtered_data, f"pharma_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    @app.route('/export_chart/<chart_type>')
    @login_required
    def export_chart(chart_type):
        # Get filter parameters
        date_range = request.args.get('date_range', 'Last 30 days')
        category = request.args.get('category', 'All Categories')
        product_id = request.args.get('product_id', '')
        format = request.args.get('format', 'png')
        
        # Load and filter data
        data = load_data()
        filtered_data = filter_data(data, date_range, category, product_id)
        
        # Generate chart based on type
        if chart_type == 'revenue':
            fig = create_interactive_revenue_chart(filtered_data)
        elif chart_type == 'inventory':
            fig = create_interactive_inventory_chart(filtered_data)
        elif chart_type == 'reservations':
            fig = create_interactive_reservations_chart(filtered_data)
        elif chart_type == 'comparison':
            fig = create_product_comparison_chart(filtered_data)
        else:
            return "Invalid chart type", 400
        
        # Export based on format
        if format == 'png':
            return export_chart_to_image(fig, f"{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        elif format == 'html':
            return export_chart_to_html(fig, f"{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        else:
            return "Invalid format", 400

    @app.route('/delete-file/<int:file_id>', methods=['POST'])
    @login_required
    def delete_file(file_id):
        try:
            # Get the file record
            file_record = UploadedFile.query.get_or_404(file_id)
            
            # Delete the actual file if it exists
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_record.filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    app.logger.warning(f'Error deleting file {file_path}: {str(e)}')
            
            # Delete model files and cached data
            try:
                # Clear model files
                models_dir = os.path.join('FLASK MODELS')
                models_path = os.path.join(models_dir, 'pharma_forecasting_dataset.csv')
                if os.path.exists(models_path):
                    os.remove(models_path)
                
                # Clear saved models directory
                saved_models_dir = os.path.join(models_dir, 'saved_models')
                if os.path.exists(saved_models_dir):
                    for model_file in os.listdir(saved_models_dir):
                        try:
                            os.remove(os.path.join(saved_models_dir, model_file))
                        except Exception as e:
                            app.logger.warning(f'Error deleting model file {model_file}: {str(e)}')
                
                # Clear any cached data
                if hasattr(app, 'cached_data'):
                    delattr(app, 'cached_data')
            except Exception as e:
                app.logger.warning(f'Error cleaning up model files: {str(e)}')
            
            # Delete all associated data from database
            try:
                # Use SQLAlchemy text() for raw SQL queries
                db.session.execute(
                    text('DELETE FROM product_forecast WHERE source_file_id = :file_id'),
                    {'file_id': file_id}
                )
                db.session.execute(
                    text('DELETE FROM inventory_prediction WHERE source_file_id = :file_id'),
                    {'file_id': file_id}
                )
                db.session.execute(
                    text('DELETE FROM model_metric WHERE source_file_id = :file_id'),
                    {'file_id': file_id}
                )
                
                # Delete the file record itself
                db.session.delete(file_record)
                db.session.commit()
                
                # Reset models and clear caches
                from .utils.model_processor import reset_models
                reset_models()
                
                # Clear any existing forecasts
                ProductForecast.query.filter_by(source_file_id=file_id).delete()
                InventoryPrediction.query.filter_by(source_file_id=file_id).delete()
                ModelMetric.query.filter_by(source_file_id=file_id).delete()
                db.session.commit()
                
                # Emit socket event to refresh client-side graphs
                if socketio:
                    socketio.emit('refresh_graphs', {'status': 'success'})
                
                flash('File and associated data deleted successfully', 'success')
                
            except Exception as e:
                db.session.rollback()
                app.logger.error(f'Error deleting database records: {str(e)}')
                flash(f'Error deleting database records: {str(e)}', 'danger')
                raise
            
        except Exception as e:
            app.logger.error(f'Error deleting file: {str(e)}')
            flash(f'Error deleting file: {str(e)}', 'danger')
            db.session.rollback()
        
        return redirect(url_for('upload_history'))

    # Socket.IO events
    @socketio.on('connect')
    def handle_connect():
        if not current_user.is_authenticated:
            return False
        print('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    @socketio.on('request_update')
    def handle_update_request(data):
        if not current_user.is_authenticated:
            return
        # Get filter parameters
        product_id = data.get('product_id', '')
        
        # Load and filter data
        data = load_data()
        filtered_data = filter_data(data, 'Last 7 days', 'All Categories', product_id)
        
        # Generate visualizations
        charts = generate_all_visualizations(filtered_data)
        
        # Convert charts to HTML strings
        chart_html = {
            'revenue': charts['revenue'],
            'inventory': charts['inventory'],
            'reservations': charts['reservations'],
            'comparison': charts['comparison']
        }
        
        # Emit update event
        emit('update_charts', chart_html)

    return app

if __name__ == '__main__':
    app = create_app()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
