from app import create_app, db
from app.models import User

def init_db():
    app = create_app()
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if admin user exists
        admin_user = User.query.filter_by(email='test@sample.com').first()
        if not admin_user:
            # Create admin user with hashed password
            admin_user = User(
                email='test@sample.com',
                is_admin=True
            )
            admin_user.set_password('test123!')
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created successfully!")
        else:
            print("Admin user already exists!")

if __name__ == '__main__':
    init_db() 