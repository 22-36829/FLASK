@echo off
echo Setting up Flask Dashboard environment...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.9 or later.
    exit /b 1
)

:: Check if virtual environment exists, create if it doesn't
if not exist "env" (
    echo Creating virtual environment...
    python -m venv env
)

:: Activate virtual environment
call env\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
echo Installing required packages...
pip install -r requirements.txt

:: Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "FLASK MODELS" mkdir "FLASK MODELS"
if not exist "instance" mkdir instance

:: Initialize the database
echo Initializing database...
python -c "from app import create_app, db; app=create_app(); app.app_context().push(); db.create_all()"

echo Setup complete! Run 'python app.py' to start the application.
pause 