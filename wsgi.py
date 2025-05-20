"""
WSGI entry point for the application.
This file ensures eventlet monkey patching happens before any other imports.
"""

# Monkey patch before ANY other imports
import eventlet
eventlet.monkey_patch(
    os=True,
    socket=True,
    time=True,
    select=True,
    thread=True,
    all=False
)

# Configure eventlet
import eventlet.debug
eventlet.debug.hub_prevent_multiple_readers(False)
eventlet.debug.hub_exceptions(True)

# Now we can safely import the rest
from main import app, socketio

# The application is already wrapped by SocketIO
application = app 