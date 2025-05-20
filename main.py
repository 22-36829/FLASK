import os
from app import create_app
from flask_socketio import SocketIO

# Create the Flask application instance
app = create_app()

# Initialize SocketIO with the correct configuration for production
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

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
    application = app  # For WSGI servers
    # Make socketio instance available for gunicorn
    # The socketio instance is what Gunicorn needs to serve 