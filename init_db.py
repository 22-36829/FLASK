from app import create_app, db
from app.models import User
from flask_bcrypt import Bcrypt

def init_db():
    app = create_app()
    bcrypt = Bcrypt(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if admin user exists
        admin_user = User.query.filter_by(email='test@sample.com').first()
        if not admin_user:
            # Create admin user with bcrypt hashed password
            hashed_password = bcrypt.generate_password_hash('test123!').decode('utf-8')
            admin_user = User(
                email='test@sample.com',
                password=hashed_password,
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created successfully!")
        else:
            print("Admin user already exists!")

if __name__ == '__main__':
    init_db() 