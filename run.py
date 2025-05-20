from app import create_app
import webbrowser
import threading

app = create_app()
print("App created!")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    print("=" * 80)
    print("Pharma Analytics Dashboard")
    print("=" * 80)
    print("Starting Flask server...")
    print("Access the dashboard at: http://localhost:5000/pharma_dashboard")
    print("=" * 80)
    # Open browser after 1 second delay
    threading.Timer(1, open_browser).start()
    # Run Flask app
    app.run(debug=True, use_reloader=False)
