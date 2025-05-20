# Flask Pharmaceutical Dashboard

A real-time pharmaceutical dashboard built with Flask, featuring WebSocket support for live updates and data visualization.

## Local Setup Instructions

### Prerequisites
- Python 3.11.0 or higher
- Git
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/22-36829/FLASK.git
cd FLASK
```

### Step 2: Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
Create a `.env` file in the root directory with the following content:
```env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
DATABASE_URL=sqlite:///instance/dashboard.db
```

### Step 5: Initialize Database
```bash
flask db upgrade
```

### Step 6: Run the Application
```bash
python app.py
```

The application will be available at: `http://localhost:10000`

## Project Structure
```
flask_dashboard/
├── app/                    # Application package
│   ├── __init__.py        # App initialization
│   ├── models/            # Database models
│   ├── routes/            # Route handlers
│   ├── static/            # Static files (CSS, JS)
│   └── templates/         # HTML templates
├── instance/              # Instance-specific files
├── uploads/              # File upload directory
├── logs/                 # Application logs
├── requirements.txt      # Project dependencies
├── app.py               # Application entry point
└── render.yaml          # Render deployment config
```

## Features
- Real-time data visualization
- WebSocket support for live updates
- File upload capabilities
- PostgreSQL database support (production)
- SQLite database (development)
- Secure authentication system
- Interactive dashboards

## Troubleshooting

### Common Issues
1. **Port already in use**
   - Change the port in `app.py` or kill the process using the current port

2. **Database connection issues**
   - Ensure SQLite is properly initialized for development
   - Check database URL in `.env` file

3. **Missing dependencies**
   - Run `pip install -r requirements.txt` again
   - Ensure virtual environment is activated

### Getting Help
If you encounter any issues, please:
1. Check the logs in the `logs/` directory
2. Create an issue on the GitHub repository
3. Ensure all prerequisites are properly installed

## License
This project is licensed under the MIT License - see the LICENSE file for details. 