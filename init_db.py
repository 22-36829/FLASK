from app import create_app, db
from app.models import User
import os
import sys
from sqlalchemy.exc import OperationalError
import time

def init_db():
    app = create_app()
    
    # Maximum number of retries for database connection
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            with app.app_context():
                # Create all tables
                db.create_all()
                
                # Only create test admin in development
                if os.environ.get('FLASK_ENV') != 'production':
                    # Check if admin user exists
                    admin_user = User.query.filter_by(email='test@sample.com').first()
                    if not admin_user:
                        # Create admin user with hashed password
                        admin_user = User(
                            email='test@sample.com',
                            is_admin=True
                        )
                        bcrypt = app.extensions['bcrypt']
                        admin_user.set_password('test123!', bcrypt)
                        db.session.add(admin_user)
                        try:
                            db.session.commit()
                            print("Admin user created successfully!")
                        except Exception as e:
                            db.session.rollback()
                            print(f"Error creating admin user: {str(e)}")
                    else:
                        print("Admin user already exists!")
                
                print("Database initialized successfully!")
                break
                
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Database connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                sys.exit(1)
        except Exception as e:
            print(f"Unexpected error initializing database: {str(e)}")
            sys.exit(1)

if __name__ == '__main__':
    # Set environment variables for local development if not set
    if not os.environ.get('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'development'
    
    init_db() 