import eventlet
eventlet.monkey_patch()  # This must be the first import

import os
from app import create_app
from flask_socketio import SocketIO

# Create the Flask application instance
app = create_app()
app.app_context().push()  # Push an application context

# Initialize SocketIO with the correct configuration for production
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=True, engineio_logger=True)

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