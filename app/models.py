from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import logging

db = SQLAlchemy()

# Configure logging once
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# User table
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.email}>'

    def set_password(self, password, bcrypt):
        """Hash password before storing"""
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password, bcrypt):
        """Check if provided password matches hash"""
        if not self.password:
            return False
        return bcrypt.check_password_hash(self.password, password)

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

# Inventory table
class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Integer, default=0)
    price = db.Column(db.Float, nullable=False)

    def __init__(self, item_name, category, quantity, price):
        self.item_name = item_name
        self.category = category
        self.quantity = quantity
        self.price = price
        logging.info(f'Inventory item created: {item_name} - {category} - Qty: {quantity} - Price: {price}')

    def __repr__(self):
        return f"<Inventory {self.item_name}>"

# Sale table
class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False)
    quantity_sold = db.Column(db.Integer, nullable=False)
    sale_date = db.Column(db.DateTime, default=datetime.utcnow)

    inventory = db.relationship('Inventory', backref=db.backref('sales', lazy=True))

    def __init__(self, item_id, quantity_sold):
        self.item_id = item_id
        self.quantity_sold = quantity_sold
        logging.info(f'Sale created: Item ID = {item_id}, Quantity Sold = {quantity_sold}')

    def __repr__(self):
        return f"<Sale {self.id} of item {self.item_id}>"

# Uploaded Files table
class UploadedFile(db.Model):
    """Model for uploaded data files"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='pending')
    error_message = db.Column(db.Text)
    
    # Relationships
    forecasts = db.relationship('ProductForecast', backref='source_file', lazy=True)
    predictions = db.relationship('InventoryPrediction', backref='source_file', lazy=True)
    metrics = db.relationship('ModelMetric', backref='source_file', lazy=True)

    def __repr__(self):
        return f'<UploadedFile {self.filename}>'

# Product Forecasts table
class ProductForecast(db.Model):
    """Model for product forecasts"""
    id = db.Column(db.Integer, primary_key=True)
    source_file_id = db.Column(db.Integer, db.ForeignKey('uploaded_file.id'), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    forecast_type = db.Column(db.String(50), nullable=False)  # revenue, sales, reservations
    forecast_date = db.Column(db.DateTime, nullable=False)
    forecast_value = db.Column(db.Float, nullable=False)
    confidence_lower = db.Column(db.Float)
    confidence_upper = db.Column(db.Float)
    model_version = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ProductForecast {self.product_id} - {self.forecast_type}>'

# Inventory Predictions table
class InventoryPrediction(db.Model):
    """Model for inventory predictions"""
    id = db.Column(db.Integer, primary_key=True)
    source_file_id = db.Column(db.Integer, db.ForeignKey('uploaded_file.id'), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False)
    days_to_zero = db.Column(db.Integer)
    reorder_point = db.Column(db.Float)
    optimal_order_qty = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<InventoryPrediction {self.product_id}>'

# Model Metrics table
class ModelMetric(db.Model):
    """Model for tracking model performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    source_file_id = db.Column(db.Integer, db.ForeignKey('uploaded_file.id'), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # arima_sales, xgb_inventory, prophet_resv
    mape = db.Column(db.Float)  # Mean Absolute Percentage Error
    rmse = db.Column(db.Float)  # Root Mean Square Error
    mae = db.Column(db.Float)   # Mean Absolute Error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelMetric {self.product_id} - {self.model_type}>'

# Legacy FileUpload model (kept for backward compatibility)
class FileUpload(db.Model):
    """Model for tracking file uploads and processing"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='Success')
    forecasts_generated = db.Column(db.Integer, default=0)
    
    def __repr__(self):
        return f'<FileUpload {self.filename}>'
