# Deployment Guide for Pharma Analytics Dashboard

This document provides detailed instructions for deploying the Pharma Analytics Dashboard to different environments.

## Local Deployment

### Prerequisites
- Python 3.9+ installed
- Git installed (optional)

### Steps

1. Clone or download the repository:
```
git clone https://github.com/yourusername/pharma-analytics.git
cd pharma-analytics
```

2. Create and activate a virtual environment:
```
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Run the application:
```
python run.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Deployment to Render

Render is a cloud platform that makes it easy to deploy web applications.

### Prerequisites
- A Render account (sign up at https://render.com)
- Your project in a Git repository (GitHub, GitLab, etc.)

### Steps

1. Log in to your Render account

2. Click on "New" and select "Web Service"

3. Connect your GitHub/GitLab account and select the repository

4. Configure the web service:
   - Name: `pharma-analytics` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn "app:create_app()" --bind=0.0.0.0:$PORT`
   - Select the appropriate plan (Free tier works for demonstration)

5. Click "Create Web Service"

6. Your application will be deployed and available at the URL provided by Render

## Deployment to Heroku

### Prerequisites
- A Heroku account (sign up at https://heroku.com)
- Heroku CLI installed
- Git installed

### Steps

1. Create a `Procfile` in the root directory with the following content:
```
web: gunicorn "app:create_app()" --log-file -
```

2. Log in to Heroku CLI:
```
heroku login
```

3. Create a new Heroku app:
```
heroku create pharma-analytics
```

4. Deploy your application:
```
git push heroku main
```

5. Open the application:
```
heroku open
```

## Environment Variables

For production deployments, consider setting the following environment variables:

- `SECRET_KEY`: A secure secret key for your Flask application
- `DATABASE_URL`: If using a different database than SQLite
- `FLASK_ENV`: Set to `production` for production deployments

## Database Migration

If you're using a different database in production:

1. Update the `SQLALCHEMY_DATABASE_URI` in `app/__init__.py`
2. Run database migrations:
```
flask db init
flask db migrate
flask db upgrade
```

## Troubleshooting

### Common Issues

1. **Application Error on Render/Heroku**:
   - Check the logs: `heroku logs --tail` or Render dashboard
   - Ensure all dependencies are in requirements.txt
   - Verify the Procfile is correctly formatted

2. **Database Connection Issues**:
   - Check database connection string
   - Ensure database service is running
   - Verify network permissions

3. **Static Files Not Loading**:
   - Check that static files are properly configured
   - Verify file paths in templates

For additional help, refer to the Flask documentation or contact the project maintainer. 