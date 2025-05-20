from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from urllib.parse import urlparse
from .models import User, db
from .forms import LoginForm
import logging

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(email=form.email.data).first()
            if user is None:
                logging.warning(f"Login attempt failed: User not found - {form.email.data}")
                flash('Invalid email or password', 'danger')
                return redirect(url_for('auth.login'))
            
            bcrypt = current_app.extensions['bcrypt']
            if not user.check_password(form.password.data, bcrypt):
                logging.warning(f"Login attempt failed: Invalid password for user - {form.email.data}")
                flash('Invalid email or password', 'danger')
                return redirect(url_for('auth.login'))
            
            login_user(user)
            logging.info(f"User logged in successfully: {user.email}")
            
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('dashboard')
            return redirect(next_page)
            
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            db.session.rollback()
            flash('An error occurred during login. Please try again.', 'danger')
            return redirect(url_for('auth.login'))
    
    return render_template('login.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    try:
        user_email = current_user.email
        logout_user()
        logging.info(f"User logged out successfully: {user_email}")
        flash('You have been logged out successfully.', 'success')
    except Exception as e:
        logging.error(f"Logout error: {str(e)}")
        flash('An error occurred during logout.', 'warning')
    
    return redirect(url_for('auth.login')) 