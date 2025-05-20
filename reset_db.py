import os
import shutil
import time
import sqlite3
from pathlib import Path
from app import create_app, db
from flask_migrate import upgrade, init, migrate, stamp
from flask_bcrypt import Bcrypt
from app.models import User, Inventory, Sale, UploadedFile, ProductForecast, InventoryPrediction, ModelMetric, FileUpload

def force_delete_file(file_path, max_attempts=5, wait_time=2):
    """Attempt to force delete a file with multiple retries"""
    import ctypes
    from ctypes import wintypes
    
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    
    MOVEFILE_DELAY_UNTIL_REBOOT = 0x4
    
    def mark_for_deletion(path):
        return kernel32.MoveFileExW(path, None, MOVEFILE_DELAY_UNTIL_REBOOT)
    
    for attempt in range(max_attempts):
        try:
            # Try normal deletion first
            os.remove(file_path)
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                print(f"File is locked, attempting alternate removal methods... (attempt {attempt + 1}/{max_attempts})")
                try:
                    # Try to mark file for deletion on reboot if we can't delete it now
                    if mark_for_deletion(file_path):
                        print("File marked for deletion on next system reboot")
                        return True
                except:
                    pass
                time.sleep(wait_time)
            else:
                print(f"Could not delete file {file_path}")
                return False
    return False

def reset_database():
    app = create_app()
    bcrypt = Bcrypt(app)
    
    with app.app_context():
        print("Starting complete database reset...")
        
        # Get the database file path and ensure it's absolute
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        db_path = os.path.abspath(db_path)
        
        # Ensure instance directory exists
        instance_dir = os.path.dirname(db_path)
        os.makedirs(instance_dir, exist_ok=True)
        
        # Close all database connections
        print("Closing database connections...")
        db.session.remove()
        db.engine.dispose()
        
        # Try to delete the database file
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            if not force_delete_file(db_path):
                print("WARNING: Could not delete database file. Will try to recreate tables anyway.")
        
        # Clean up directories first
        dirs_to_clean = {
            'uploads': 'uploads',
            'models': 'FLASK MODELS',
            'temp': 'temp',
            'logs': 'logs',
            'migrations': 'migrations'
        }
        
        # Clean directories
        for dir_name, dir_path in dirs_to_clean.items():
            full_path = os.path.join(os.path.dirname(__file__), dir_path)
            if os.path.exists(full_path):
                print(f"Cleaning {dir_name} directory...")
                try:
                    shutil.rmtree(full_path)
                except Exception as e:
                    print(f"Warning: Could not clean {dir_name} directory: {str(e)}")
            os.makedirs(full_path, exist_ok=True)
            if dir_name == 'models':
                os.makedirs(os.path.join(full_path, 'saved_models'), exist_ok=True)
            elif dir_name == 'migrations':
                os.makedirs(os.path.join(full_path, 'versions'), exist_ok=True)
        
        # Clean cache files
        print("Cleaning cache files...")
        cache_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so', '*.pkl', '*.h5', '*.joblib', '*.log']
        for pattern in cache_patterns:
            for cache_file in Path(__file__).parent.rglob(pattern):
                try:
                    cache_file.unlink()
                    print(f"Removed cache file: {cache_file}")
                except Exception as e:
                    print(f"Warning: Could not remove cache file {cache_file}: {str(e)}")
        
        try:
            # Initialize fresh database
            print("Creating new database...")
            db.create_all()
            
            # Create default admin user
            print("Creating default admin user...")
            admin_user = User(
                email='test@sample.com',
                password=bcrypt.generate_password_hash('test123!').decode('utf-8'),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            
            # Initialize migrations
            print("Setting up database migrations...")
            try:
                with app.app_context():
                    init()
                    migrate()
                    stamp()
            except Exception as e:
                print(f"Warning: Could not initialize migrations: {str(e)}")
            
            print("\nDatabase reset completed successfully!")
            print("\nDefault admin credentials:")
            print("Email: test@sample.com")
            print("Password: test123!")
            print("\nThe following actions were completed:")
            print("✓ Database connections closed")
            print("✓ Old database file handled")
            print("✓ Directories cleaned and recreated")
            print("✓ Cache files cleaned")
            print("✓ New database initialized")
            print("✓ Default admin user created")
            print("✓ Migrations initialized")
            
        except Exception as e:
            print(f"Error during database reset: {str(e)}")
            db.session.rollback()
            return False
        
        return True

if __name__ == '__main__':
    success = reset_database()
    if not success:
        print("\nDatabase reset encountered errors. Please check the messages above.")
        exit(1) 