import eventlet
eventlet.monkey_patch()  # This must be the first import

import os
from app import create_app
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

# Create the Flask application instance
app = create_app()
app.app_context().push()  # Push an application context

# Configure SQLAlchemy for thread-safety with eventlet
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 60,
    'pool_pre_ping': True,
    'pool_use_lifo': True
}

# Initialize SocketIO with the correct configuration for production
socketio = SocketIO(app, 
                   async_mode='eventlet', 
                   cors_allowed_origins="*", 
                   logger=True, 
                   engineio_logger=True,
                   manage_session=False)  # Let Flask-SQLAlchemy handle sessions

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 10000))
    print("\nServer starting...")
    print(f"Access the dashboard at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server\n")
    socketio.run(app, debug=True, host='0.0.0.0', port=port)
else:
    # This is for Gunicorn/production
    port = int(os.environ.get('PORT', 10000))
    application = socketio.middleware(app)  # Use socketio middleware 