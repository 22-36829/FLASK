from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from functools import wraps
import logging
import time
from contextlib import contextmanager

db = SQLAlchemy()
logger = logging.getLogger(__name__)

def retry_on_deadlock(f):
    """Decorator to retry database operations on deadlock"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 0.1  # seconds
        
        for attempt in range(max_retries):
            try:
                return f(*args, **kwargs)
            except OperationalError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for database operation: {str(e)}")
                    raise
                logger.warning(f"Database deadlock detected, retrying... (attempt {attempt + 1})")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                db.session.rollback()
        return None
    return wrapper

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = db.session
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database error occurred: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

def check_connection():
    """Check database connection and report status"""
    try:
        # Try a simple query
        db.session.execute("SELECT 1")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

def cleanup_connections():
    """Clean up database connections"""
    try:
        db.session.remove()
        db.engine.dispose()
    except Exception as e:
        logger.error(f"Error cleaning up database connections: {str(e)}")

def init_db(app):
    """Initialize database with enhanced error handling"""
    try:
        db.init_app(app)
        
        # Register connection cleanup
        @app.teardown_appcontext
        def shutdown_session(exception=None):
            cleanup_connections()
        
        # Add health check endpoint
        @app.route('/health/db')
        def db_health_check():
            if check_connection():
                return {'status': 'healthy', 'message': 'Database connection is active'}, 200
            return {'status': 'unhealthy', 'message': 'Database connection failed'}, 503
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise 