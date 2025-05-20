"""
WSGI entry point for the application.
This file ensures eventlet monkey patching happens before any other imports.
"""
import eventlet
eventlet.monkey_patch(all=True, thread=True, select=True)

# Configure eventlet
import eventlet.debug
eventlet.debug.hub_prevent_multiple_readers(False)
eventlet.debug.hub_exceptions(True)

# Now we can safely import the rest
from main import application, socketio

# The WSGI application is already properly configured in main.py
app = application 