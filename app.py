from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import base64
import os
import multiprocessing
from datetime import datetime
from utils import database, enhanced_face_recognition as face_recognition, auth
from functools import lru_cache, wraps
import threading
import time
import numpy as np
import cv2
import sys
import logging
import traceback
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure value

# Add startup message
logger.info("Starting Flask application...")

# Initialize databases
database.initialize_db()
auth.initialize_admins()

# Synchronize employees with reference folders
def sync_employees_with_references():
    """Synchronize employees database with reference folders"""
    # Get list of existing employees
    employees = database.get_employees()
    existing_ids = {emp['id'] for emp in employees}
    
    # Get list of reference folders
    reference_path = 'static/references'
    if not os.path.exists(reference_path):
        return
    
    # Get all employee folders (excluding files and system directories)
    employee_folders = []
    for item in os.listdir(reference_path):
        folder_path = os.path.join(reference_path, item)
        if os.path.isdir(folder_path) and not item.startswith('.') and not item.startswith('_'):
            # Skip any system files or model directories
            if not item.startswith('ds_model_') and not item.endswith('.pkl'):
                employee_folders.append(item)
    
    # Find missing employee IDs
    missing_ids = []
    for employee_id in employee_folders:
        if employee_id not in existing_ids and employee_id.strip():
            missing_ids.append(employee_id)
    
    if missing_ids:
        print(f"Found {len(missing_ids)} employees in references but missing from database. Adding them...")
        
        # Add missing employees to the database
        for employee_id in missing_ids:
            try:
                # Add the employee with default values
                database.add_employee(
                    employee_id=employee_id,
                    name=employee_id,  # Just use ID directly as name
                    status="Active"
                )
                print(f"Added missing employee {employee_id} to database")
            except Exception as e:
                print(f"Error adding employee {employee_id}: {e}")

# Run synchronization before loading reference images
sync_employees_with_references()

# Optimize thread pool size based on available CPU cores
cpu_count = os.cpu_count() or 4
max_workers = min(8, cpu_count)
os.environ['TF_NUM_INTEROP_THREADS'] = str(max_workers)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(max_workers)

# Load reference images for face recognition
reference_images, reference_names = face_recognition.load_reference_images()
print(f"Loaded {len(reference_images)} reference images on startup")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('user_type') != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    
    if request.method == 'POST':
        # Verify login credentials
        admin_id = request.form['admin_id']
        password = request.form['password']
        
        if auth.verify_login(admin_id, password):
            # Set session
            session['user_id'] = admin_id
            session['user_type'] = 'admin'
            return redirect(url_for('home'))
        else:
            error = 'Invalid admin ID or password'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    # Clear session
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
@admin_required
def home():
    # Get stats for dashboard
    employees = database.get_employees()
    active_employees = sum(1 for emp in employees if emp['status'] == 'Active')
    
    # Get today's access records
    records = database.get_access_records()
    today = datetime.now().strftime("%Y-%m-%d")
    
    access_granted = sum(1 for rec in records if rec['access'] == 'Granted' and rec['timestamp'].startswith(today))
    access_denied = sum(1 for rec in records if rec['access'] == 'Denied' and rec['timestamp'].startswith(today))
    
    return render_template('home.html', 
                           employees=employees, 
                           active_employees=active_employees,
                           access_granted=access_granted,
                           access_denied=access_denied)

@app.route('/report')
@admin_required
def report():
    # Get all access records
    records = database.get_access_records()
    
    # Sort records by timestamp (newest first)
    records.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('report.html', records=records)

@app.route('/employee')
@admin_required
def employee():
    # Get all employees
    employees = database.get_employees()
    
    return render_template('employee.html', employees=employees)

@app.route('/add_employee')
@admin_required
def add_employee_page():
    return render_template('add_employee.html')

@app.route('/add_employee', methods=['POST'])
@admin_required
def add_new_employee_with_face():
    """Add new employee with enhanced reference images"""
    global reference_images, reference_names
    
    # Get form data
    employee_id = request.form['employee_id'].strip()  # Trim whitespace
    name = request.form['name']
    status = request.form['status']
    specs_mode = request.form.get('specs_mode', 'without')
    
    # Check if employee ID already exists
    existing_employee = database.get_employee(employee_id)
    if existing_employee:
        flash(f'Employee ID {employee_id} already exists. Please use a different ID.', 'error')
        return redirect(url_for('add_employee_page'))
    
    # Get all angle images for primary set
    angle_images = {
        'front': request.form.get('captured_image_front', ''),
        'left': request.form.get('captured_image_left', ''),
        'right': request.form.get('captured_image_right', ''),
        'up': request.form.get('captured_image_up', ''),
        'down': request.form.get('captured_image_down', '')
    }
    
    # Get all angle images for glasses variation (if applicable)
    glasses_variation_images = {}
    if specs_mode == 'both':
        glasses_variation_images = {
            'front_glasses': request.form.get('captured_image_front_glasses', ''),
            'left_glasses': request.form.get('captured_image_left_glasses', ''),
            'right_glasses': request.form.get('captured_image_right_glasses', ''),
            'up_glasses': request.form.get('captured_image_up_glasses', ''),
            'down_glasses': request.form.get('captured_image_down_glasses', '')
        }
    
    # Verify that we have all required angles for primary set
    if not all(angle_images.values()):
        flash('Error: Missing one or more required face angles', 'error')
        return redirect(url_for('add_employee_page'))
    
    # If using both modes, verify we have all angles for second set
    if specs_mode == 'both' and not all(glasses_variation_images.values()):
        flash('Error: Missing one or more required face angles for with glasses images', 'error')
        return redirect(url_for('add_employee_page'))
    
    # Add employee to the database
    if database.add_employee(employee_id, name, status):
        # Generate reference images from all captured angles
        success, count = face_recognition.capture_multiple_angles(employee_id, angle_images)
        
        # If we also have glasses variations, add those images to the same folder
        if specs_mode == 'both' and success:
            success2, additional_count = face_recognition.capture_multiple_angles(employee_id, glasses_variation_images)
            
            if success2:
                # Reload reference images
                reference_images, reference_names = face_recognition.load_reference_images()
                
                flash(f'Employee added successfully with {count + additional_count} reference images (with and without glasses) for enhanced recognition', 'success')
            else:
                flash('Employee added but failed to save some reference images', 'warning')
        elif success:
            # Reload reference images
            reference_images, reference_names = face_recognition.load_reference_images()
            
            flash(f'Employee added successfully with {count} reference images for enhanced recognition', 'success')
        else:
            flash('Employee added but failed to save reference images', 'error')
    else:
        flash('Failed to add employee. The ID may already exist.', 'error')
    
    return redirect(url_for('employee'))

@app.route('/delete_employee/<employee_id>', methods=['POST'])
@admin_required
def delete_employee(employee_id):
    global reference_images, reference_names
    
    # Delete employee from database
    if database.delete_employee(employee_id):
        # Delete reference image
        face_recognition.delete_user(employee_id)
        
        # Reload reference images
        reference_images, reference_names = face_recognition.load_reference_images()
        
        flash('Employee deleted successfully', 'success')
    else:
        flash('Failed to delete employee', 'error')
    
    return redirect(url_for('employee'))

@app.route('/edit_employee/<employee_id>')
@admin_required
def edit_employee_page(employee_id):
    # Get employee details
    employee = database.get_employee(employee_id)
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employee'))
    
    return render_template('edit_employee.html', employee=employee)

@app.route('/edit_employee/<employee_id>', methods=['POST'])
@admin_required
def update_employee(employee_id):
    # Get form data
    name = request.form['name']
    status = request.form['status']
    
    # Update employee in database
    if database.update_employee(employee_id, name, status):
        flash('Employee updated successfully', 'success')
    else:
        flash('Failed to update employee', 'error')
    
    return redirect(url_for('employee'))

@app.route('/recognition')
@login_required
def recognition():
    return render_template('recognition.html')

# Global processing lock to prevent overloading
recognition_lock = threading.Lock()
request_queue = []

def can_process_request():
    current_time = time.time()
    # Clean old requests
    while request_queue and request_queue[0] < current_time - 2.0:  # 2-second window
        request_queue.pop(0)
    # Check if we can process a new request
    if len(request_queue) < 3:  # Allow max 3 requests in 2-second window
        request_queue.append(current_time)
        return True
    return False

@app.route('/process_face', methods=['POST'])
def process_face():
    # Rate limiting to prevent overloading
    if not can_process_request():
        return jsonify({
            'match': False,
            'id': "Rate Limited",
            'message': "Too many requests. Please try again in a moment.",
            'confidence': 0
        }), 429  # Too Many Requests
    
    # Get image data from request
    data = request.json
    image_data = data['image'].split(',')[1]
    
    # Check if there are any reference images loaded
    if not reference_images or len(reference_images) == 0:
        print("No reference images available for comparison")
        return jsonify({
            'match': False,
            'id': "No References",
            'confidence': 0
        })
    
    # Use a lock to prevent parallel processing (which can slow things down)
    with recognition_lock:
        # Process face for recognition
        result = face_recognition.process_face(image_data, reference_images)
        # Check if we got 2 or 3 values
        if len(result) == 2:
            match, employee_id = result
            confidence = 0  # Set a default confidence value
        else:
            match, employee_id, confidence = result
    
    if match:
        # Get employee info from database
        employees = database.get_employees()
        employee = next((emp for emp in employees if emp['id'] == employee_id), None)
        
        if employee:
            # Always grant access if status is Active
            if employee['status'] == 'Active':
                access = 'Granted'
                confidence_note = "Active employee - access granted"
                database.add_access_record(employee['id'], employee['name'], employee['status'], access, confidence_note)
                return jsonify({
                    'match': True,
                    'id': employee['id'],
                    'name': employee['name'],
                    'status': employee['status'],
                    'access': access,
                    'confidence': confidence,
                    'recorded': True
                })
            else:
                # If employee is inactive, deny access immediately
                access = 'Denied'
                confidence_note = "Inactive employee - access denied"
                database.add_access_record(employee['id'], employee['name'], employee['status'], access, confidence_note)
                return jsonify({
                    'match': True,
                    'id': employee['id'],
                    'name': employee['name'],
                    'status': employee['status'],
                    'access': access,
                    'confidence': confidence,
                    'recorded': True
                })
        else:
            print(f"Error: Matched employee ID {employee_id} but not found in database")
            # Handle error case
    
    # Handle no match cases - Record the access denial here
    if employee_id == "No Face":
        # Don't record "No Face" detections in the database
        return jsonify({
            'match': False,
            'id': "No Face",
            'confidence': 0,
            'recorded': True  # Prevent client from recording it
        })
    elif employee_id == "Low Confidence":
        # Record "Low Confidence" detections in the database as Unknown
        database.add_access_record("Unknown", "Unknown", "N/A", "Denied", f"Low confidence detection ({confidence:.1f}%)")
        return jsonify({
            'match': False,
            'id': "Low Confidence",
            'confidence': confidence,
            'recorded': True  # Prevent client from recording it
        })
    
    # For genuine "Unknown" person - record the access denial
    database.add_access_record("Unknown", "Unknown", "N/A", "Denied")
    
    # No match found
    return jsonify({
        'match': False,
        'id': "Unknown",
        'confidence': 0,
        'recorded': True  # Flag to indicate server already recorded this
    })

@app.route('/clear_records', methods=['POST'])
@admin_required
def clear_records():
    # Clear all access records
    if database.clear_access_records():
        flash('Access records cleared successfully', 'success')
    else:
        flash('Failed to clear access records', 'error')
    
    return redirect(url_for('report'))

@app.route('/record_access', methods=['POST'])
def record_access():
    """Record access attempt in the database"""
    data = request.json
    
    # Extract data from request
    match = data.get('match', False)
    employee_id = data.get('id', 'Unknown')
    name = data.get('name', 'Unknown')
    status = data.get('status', 'N/A')
    access = data.get('access', 'Denied')
    
    # Add the record to the database
    database.add_access_record(employee_id, name, status, access)
    
    return jsonify({'success': True})

@app.route('/change_password', methods=['POST'])
@admin_required
def change_password():
    # Get form data
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    
    # Verify current password
    if not auth.verify_login(session['user_id'], current_password):
        flash('Current password is incorrect', 'error')
        return redirect(url_for('home'))
    
    # Verify new password matches confirmation
    if new_password != confirm_password:
        flash('New passwords do not match', 'error')
        return redirect(url_for('home'))
    
    # Update password
    if auth.change_password(session['user_id'], current_password, new_password):
        flash('Password changed successfully', 'success')
    else:
        flash('Failed to change password', 'error')
    
    return redirect(url_for('home'))

@app.route('/api/detect_face', methods=['POST'])
def detect_face():
    """API endpoint to detect faces in an image"""
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            logger.error("No image data provided in request")
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = data['image']
        min_confidence = data.get('min_confidence', 0.6)  # Default to 0.6 if not specified
        
        # Validate image data format
        if not image_data.startswith('data:image/'):
            logger.error("Invalid image format provided")
            return jsonify({'error': 'Invalid image format'}), 400
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image data")
                return jsonify({'error': 'Failed to decode image'}), 400
                
            # Convert to RGB for face detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Initialize YOLO model if not already initialized
            if face_recognition.yolo_model is None:
                logger.info("Initializing YOLO model...")
                face_recognition.initialize_yolo()
            
            # Detect faces with the specified confidence threshold
            faces = face_recognition.detect_faces(img_rgb, min_confidence)
            
            if not faces:
                logger.debug("No faces detected in image")
                return jsonify({'faces': []})
                
            # Format face detection results for frontend
            result_faces = []
            for face in faces:
                try:
                    # Log the face object for debugging
                    logger.debug(f"Face object: {face}")
                    
                    # Get bbox coordinates
                    if isinstance(face, dict):
                        if 'bbox' in face:
                            bbox = face['bbox']
                            if isinstance(bbox, tuple):
                                x1, y1, x2, y2 = bbox
                            else:
                                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        else:
                            # If bbox is directly in face object
                            x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                    else:
                        # If face is a tuple of (bbox, confidence)
                        bbox, confidence = face
                        x1, y1, x2, y2 = bbox
                    
                    result_faces.append({
                        'bbox': [x1, y1, x2-x1, y2-y1],  # Convert to [x, y, width, height] format
                        'confidence': float(confidence if isinstance(face, tuple) else face.get('confidence', 1.0))
                    })
                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}")
                    logger.error(f"Face object: {face}")
                    continue
                
            logger.debug(f"Detected {len(result_faces)} faces")
            return jsonify({'faces': result_faces})
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in face detection API: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)