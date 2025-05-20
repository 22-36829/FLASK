from app import create_app
from app.models import db, User

def create_admin_user():
    app = create_app()
    with app.app_context():
        # Check if admin user already exists
        admin = User.query.filter_by(email='test@sample.com').first()
        if admin:
            print('Admin user already exists')
            return
        
        # Create admin user
        admin = User(email='test@sample.com')
        admin.set_password('test123!')
        db.session.add(admin)
        db.session.commit()
        print('Admin user created successfully')

if __name__ == '__main__':
    create_admin_user() 