import json
import os

# Admin credentials file
ADMIN_FILE = 'utils/admins.json'

def initialize_admins():
    """Initialize the admin credentials file if it doesn't exist"""
    # Create utils directory if it doesn't exist
    os.makedirs('utils', exist_ok=True)
    
    if not os.path.exists(ADMIN_FILE):
        # Default admin credentials (in a real app, use hashed passwords)
        default_admins = [
            {
                "admin_id": "admin",
                "password": "password"  # This should be hashed in a real application
            }
        ]
        
        with open(ADMIN_FILE, 'w') as f:
            json.dump(default_admins, f, indent=4)
        
        print("Initialized default admin credentials")

def verify_login(admin_id, password):
    """Verify admin login credentials"""
    # Initialize admins if needed
    if not os.path.exists(ADMIN_FILE):
        initialize_admins()
    
    try:
        with open(ADMIN_FILE, 'r') as f:
            admins = json.load(f)
            
            # Check if credentials match any admin
            for admin in admins:
                if admin["admin_id"] == admin_id and admin["password"] == password:
                    return True
            
            return False
    except Exception as e:
        print(f"Error verifying login: {e}")
        return False

def change_password(admin_id, old_password, new_password):
    """Change an admin's password"""
    try:
        with open(ADMIN_FILE, 'r+') as f:
            admins = json.load(f)
            
            # Find and update the admin's password
            for admin in admins:
                if admin["admin_id"] == admin_id and admin["password"] == old_password:
                    admin["password"] = new_password
                    
                    # Write back to the file
                    f.seek(0)
                    json.dump(admins, f, indent=4)
                    f.truncate()
                    
                    return True
            
            return False
    except Exception as e:
        print(f"Error changing password: {e}")
        return False
