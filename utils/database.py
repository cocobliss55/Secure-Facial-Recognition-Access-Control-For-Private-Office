import json
import os
import time
from datetime import datetime

# File paths for our JSON "databases"
EMPLOYEES_FILE = 'utils/employees.json'
ACCESS_RECORDS_FILE = 'utils/access_records.json'

def initialize_db():
    """Initialize the database files with empty data if they don't exist"""
    # Create utils directory if it doesn't exist
    os.makedirs('utils', exist_ok=True)
    
    # Initialize employees file with an empty list
    if not os.path.exists(EMPLOYEES_FILE):
        # Empty list instead of sample employees
        sample_employees = []
        
        with open(EMPLOYEES_FILE, 'w') as f:
            json.dump(sample_employees, f, indent=4)
    
    # Initialize access records file with an empty list
    if not os.path.exists(ACCESS_RECORDS_FILE):
        # Empty list instead of sample records
        sample_records = []
        
        with open(ACCESS_RECORDS_FILE, 'w') as f:
            json.dump(sample_records, f, indent=4)

def get_employees():
    """Get all employees from the database"""
    try:
        with open(EMPLOYEES_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def add_employee(employee_id, name, status="Active"):
    """Add a new employee to the database with sequential roll number"""
    try:
        
        employee_id = employee_id.strip()
        
        with open(EMPLOYEES_FILE, 'r+') as f:
            try:
                employees = json.load(f)
            except json.JSONDecodeError:
                employees = []
            
            # Check if employee ID already exists
            for emp in employees:
                if emp["id"].strip() == employee_id:
                    print(f"Employee ID {employee_id} already exists.")
                    return False
            
            # Next roll number is simply the count of existing employees plus 1
            next_roll = len(employees) + 1
            
            # Add the new employee
            employees.append({
                "roll_no": next_roll,
                "id": employee_id,
                "name": name,
                "status": status
            })
            
            # Write back to the file
            f.seek(0)
            json.dump(employees, f, indent=4)
            f.truncate()
            
            # Invalidate the cache
            global _employee_cache, _employee_cache_timestamp
            _employee_cache = None
            _employee_cache_timestamp = 0
            
            return True
    except Exception as e:
        print(f"Error adding employee: {e}")
        return False

def update_employee(employee_id, name=None, status=None):
    """Update an existing employee in the database"""
    try:
        # Trim whitespace from employee_id
        employee_id = employee_id.strip()
        
        with open(EMPLOYEES_FILE, 'r+') as f:
            employees = json.load(f)
            
            # Find and update the employee
            for emp in employees:
                if emp["id"].strip() == employee_id:
                    if name:
                        emp["name"] = name
                    if status:
                        emp["status"] = status
                    break
            
            # Write back to the file
            f.seek(0)
            json.dump(employees, f, indent=4)
            f.truncate()
            
            return True
    except Exception as e:
        print(f"Error updating employee: {e}")
        return False

def delete_employee(employee_id):
    """Delete an employee from the database and reorder roll numbers"""
    try:
        # Trim whitespace from employee_id
        employee_id = employee_id.strip()
        
        with open(EMPLOYEES_FILE, 'r+') as f:
            employees = json.load(f)
            
            # Filter out the employee to delete
            employees = [emp for emp in employees if emp["id"].strip() != employee_id]
            
            # Reorder roll numbers to be sequential
            for i, emp in enumerate(employees):
                emp["roll_no"] = i + 1  # Roll numbers start from 1
            
            # Write back to the file
            f.seek(0)
            json.dump(employees, f, indent=4)
            f.truncate()
            
            return True
    except Exception as e:
        print(f"Error deleting employee: {e}")
        return False

def get_access_records():
    """Get all access records from the database"""
    try:
        with open(ACCESS_RECORDS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def add_access_record(employee_id, name, status, access, notes=""):
    """Add a new access record to the database"""
    try:
        with open(ACCESS_RECORDS_FILE, 'r+') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
            
            # Add the new record
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            records.append({
                "timestamp": timestamp,
                "id": employee_id,
                "name": name,
                "status": status,
                "access": access,
                "notes": notes
            })
            
            # Write back to the file
            f.seek(0)
            json.dump(records, f, indent=4)
            f.truncate()
            
            return True
    except Exception as e:
        print(f"Error adding access record: {e}")
        return False
    

def clear_access_records():
    """Clear all access records from the database"""
    try:
        # Create an empty array for the records
        empty_records = []
        
        # Write the empty array to the file
        with open(ACCESS_RECORDS_FILE, 'w') as f:
            json.dump(empty_records, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error clearing access records: {e}")
        return False

# In database.py, add a caching mechanism for employee data
_employee_cache = None
_employee_cache_timestamp = 0

def get_employees():
    """Get all employees from the database with caching"""
    global _employee_cache, _employee_cache_timestamp
    
    # Check if cache is still valid (10 seconds cache duration)
    current_time = time.time()
    if _employee_cache is not None and current_time - _employee_cache_timestamp < 10:
        return _employee_cache
    
    try:
        with open(EMPLOYEES_FILE, 'r') as f:
            employees = json.load(f)
            _employee_cache = employees
            _employee_cache_timestamp = current_time
            return employees
    except (FileNotFoundError, json.JSONDecodeError):
        _employee_cache = []
        _employee_cache_timestamp = current_time
        return []

def get_employee(employee_id):
    """Get a single employee by ID"""
    try:
        # Use the cached employees list instead of reading the file again
        employees = get_employees()
        
        # Trim the input employee_id
        employee_id = employee_id.strip()
        
        # Find the employee with matching ID
        for emp in employees:
            if emp["id"].strip() == employee_id:
                return emp
        
        return None
    except Exception as e:
        print(f"Error getting employee: {e}")
        return None
