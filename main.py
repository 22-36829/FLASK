import os
from app import create_app, db
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

# Create the Flask application instance
app = create_app()

# Configure SQLAlchemy for thread-safety with eventlet
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 20,
    'max_overflow': 10,
    'pool_recycle': 60,
    'pool_pre_ping': True,
    'pool_use_lifo': True,
    'connect_args': {
        'connect_timeout': 10
    }
}

# Push an application context
ctx = app.app_context()
ctx.push()

# Initialize SocketIO with the correct configuration for production
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    manage_session=False,
    ping_timeout=30,
    ping_interval=15,
    max_http_buffer_size=1024 * 1024,
    async_handlers=True
)

@socketio.on_error_default
def default_error_handler(e):
    print(f'SocketIO Error: {str(e)}')
    # Close any open database sessions
    try:
        db.session.remove()
    except:
        pass

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
    
    # Create a WSGI middleware that handles database session cleanup
    def cleanup_db_session(wsgi_app):
        def middleware(environ, start_response):
            try:
                return wsgi_app(environ, start_response)
            finally:
                try:
                    db.session.remove()
                except:
                    pass
        return middleware

    # The Flask app is our WSGI application
    application = app

    # Wrap it with the cleanup middleware
    application = cleanup_db_session(application) 