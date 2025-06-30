import cv2
import os
import numpy as np
import base64
import threading
import time
from deepface import DeepFace
import logging
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageEnhance
import io
import traceback
import uuid
import json
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_recognition')

# Global variables with optimized defaults
THRESHOLD_FACENET = 0.25  # Decreased from 0.28 for more strict matching
MIN_CONFIDENCE = 70       # Increased from 65 for more confident matching
MAX_CACHE_SIZE = 200      # Increased cache size for better performance
CACHE_DURATION = 120      # Increased cache duration to 2 minutes
MAX_IMAGE_SIZE = 640      # Maximum dimension for processing (downscaling large images)

# Reference image optimization
MAX_REFERENCES_PER_ANGLE = 3  # Increased from 2 to allow more references per angle
MAX_REFERENCES_WITH_GLASSES = 15  # Increased from 10 for more reference images
MAX_REFERENCES_WITHOUT_GLASSES = 15  # Increased from 10 for more reference images
QUALITY_THRESHOLD = 0.75  # Increased quality threshold for better reference images

# Performance optimization settings
USE_THREADING = True      # Use parallel processing when possible
MAX_WORKERS = max(2, min(6, os.cpu_count() or 4))  # Increased thread pool size for better performance

# Caching variables
recognition_cache = {}
recognition_cache_timestamps = {}
yolo_model = None  # Global YOLO model instance
model_initialized = False  # Flag to check if model is already initializing

# Dictionary to store reference images
user_reference_images = {}

# Locks for thread safety
cache_lock = threading.Lock()
model_init_lock = threading.Lock()  # Lock for model initialization

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

#######################################################
# SECTION 1: FACE DETECTION - CORE FUNCTIONS
#######################################################

def initialize_yolo():
    """Initialize the YOLO model"""
    global yolo_model, model_initialized
    
    with model_init_lock:
        if yolo_model is None and not model_initialized:
            model_initialized = True  
            try:
                yolo_model = YOLO('models/best.pt')
                # Set model parameters for faster inference
                yolo_model.conf = 0.5  
                yolo_model.iou = 0.3   
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                model_initialized = False  
                logger.error(f"Error loading YOLO model: {e}")
                raise

def detect_faces(image, confidence_threshold=0.5):
    """Detect faces in an image using YOLO"""
    global yolo_model
    
    if yolo_model is None:
        initialize_yolo()
    
    try:
        h, w = image.shape[:2]
        if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE:
            image = resize_if_large(image, MAX_IMAGE_SIZE)
        
        # Run YOLO inference with specified confidence threshold
        results = yolo_model(image, 
                           conf=confidence_threshold,
                           iou=0.3,            
                           max_det=10,         
                           verbose=False)      
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                faces.append(((x, y, w, h), float(confidence)))
        
        return faces
    except Exception as e:
        logger.error(f"Error in YOLO face detection: {e}")
        return []

#######################################################
# SECTION 2: FACE RECOGNITION - CORE FUNCTIONS
#######################################################

def process_face(image_data, reference_images):
    """Optimized face recognition using Facenet for better accuracy and speed"""
    global recognition_cache, recognition_cache_timestamps
    
    try:
        # Check cache first for speed
        cache_key = hash(image_data[:100] + image_data[-100:])
        
        with cache_lock:
            cached_result = recognition_cache.get(cache_key)
            cached_timestamp = recognition_cache_timestamps.get(cache_key, 0)
            
            # Check if cache is still valid
            if cached_result and (time.time() - cached_timestamp) < CACHE_DURATION:
                return cached_result
        
        # No reference images - immediately return no match
        if not reference_images or len(reference_images) == 0:
            logger.warning("No reference images available for comparison")
            return False, "No References", 0
        
        start_time = time.time()
        
        # Decode image
        img_data = base64.b64decode(image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode input image")
            return False, "Invalid Image", 0
            
        # Resize large images for faster processing
        frame = resize_if_large(frame, MAX_IMAGE_SIZE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        # Multi-stage face detection - try high quality first, then fallback to lower thresholds
        detected_faces = detect_faces(frame_rgb, 0.8)  # Higher threshold for better quality
        
        # If no faces found with the high-quality method, try medium threshold
        if not detected_faces:
            logger.info("High-quality face detection failed, trying with medium confidence")
            detected_faces = detect_faces(frame_rgb, 0.6)
            
        # If still no faces found, try with very permissive threshold
        if not detected_faces:
            logger.info("Medium-quality face detection failed, trying with low confidence")
            detected_faces = detect_faces(frame_rgb, 0.4)
        
        # If still no faces found, return no face
        if not detected_faces:
            logger.warning("No faces detected by any method, returning No Face")
            return False, "No Face", 0
        
        # Get largest face
        largest_face = max(detected_faces, key=lambda face: face[0][2] * face[0][3])
        x, y, w, h = largest_face[0]
        face_region = frame_rgb[y:y+h, x:x+w]
        
        # Calculate face size for dynamic threshold adjustment
        face_size_percentage = (w * h) / (frame_rgb.shape[0] * frame_rgb.shape[1]) * 100
        
        # Adjust threshold based on face size and quality
        facenet_threshold = THRESHOLD_FACENET
        min_confidence = MIN_CONFIDENCE
        
        # Adaptive threshold - tighter for large clear faces, more lenient for small faces
        if face_size_percentage < 5:  
            logger.info(f"Small face detected ({face_size_percentage:.2f}%), adjusting threshold")
            facenet_threshold *= 1.2  
            min_confidence -= 5  
        elif face_size_percentage > 20:  
            # Large, clear faces should have stricter matching requirements
            facenet_threshold *= 0.9
            min_confidence += 5
            
        # Check if the face is too blurry - estimate blurriness using Laplacian variance
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if variance < 50:  # Very blurry face
                logger.info(f"Blurry face detected (variance: {variance:.2f}), adjusting threshold")
                facenet_threshold *= 1.2
        except Exception as e:
            logger.error(f"Error in blur detection: {e}")
        
        # Process with Facenet model
        try:
            # Save the face temporarily as a file for DeepFace.find
            temp_face_path = "static/cache/temp_face.jpg"
            os.makedirs("static/cache", exist_ok=True)
            cv2.imwrite(temp_face_path, cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))
            
            # Use DeepFace.find with optimized parameters - use both Facenet and VGG-Face models for verification
            # First try with primary model (Facenet)
            result = DeepFace.find(
                img_path=temp_face_path,
                db_path="static/references",
                model_name="Facenet",
                distance_metric="cosine",
                enforce_detection=False,
                threshold=facenet_threshold,
                silent=True
            )
            
            # Process the results
            if not isinstance(result, list) or (isinstance(result, list) and len(result) > 0 and not result[0].empty):
                # Convert to DataFrame if it's a list of DataFrames
                if isinstance(result, list):
                    result = result[0]
                    
                if result.empty:
                    logger.warning("DeepFace returned empty result")
                    return False, (False, "Unknown", 0)
                    
                # First result is the most similar face
                best_match = result.iloc[0]
                identity_path = best_match['identity']
                
                # Extract user ID from the folder path
                path_parts = identity_path.split(os.sep)
                user_id = path_parts[-2]  # Second to last element should be employee_id
                
                # Calculate confidence based on distance and cap at 95%
                distance = best_match['distance']
                confidence = min(95.0, (1 - min(distance, 1.0)) * 100)
                
                # Enhanced verification for more accuracy - check if we have multiple close matches
                multiple_matches = False
                if len(result) > 1:
                    second_best = result.iloc[1]
                    second_distance = second_best['distance']
                    second_identity_path = second_best['identity']
                    second_path_parts = second_identity_path.split(os.sep)
                    second_user_id = second_path_parts[-2]
                    
                    # If two top matches are from different people and very close in score
                    if second_user_id != user_id and abs(distance - second_distance) < 0.2:
                        multiple_matches = True
                        logger.warning(f"Multiple close matches detected: {user_id} ({distance:.4f}) vs {second_user_id} ({second_distance:.4f})")
                        
                        # Verify with a second model for confirmation
                        try:
                            verify_result = DeepFace.verify(
                                img1_path=temp_face_path,
                                img2_path=identity_path,
                                model_name="VGG-Face",  # Use different model for verification
                                enforce_detection=False,
                                distance_metric="cosine"
                            )
                            
                            if verify_result["verified"]:
                                logger.info(f"Second model confirmed the match for {user_id}")
                                # Boost confidence since we have multi-model confirmation
                                confidence = min(95.0, confidence + 5)
                            else:
                                logger.warning(f"Second model rejected the match for {user_id}")
                                # Reduce confidence due to conflict between models
                                confidence = max(0.0, confidence - 10)
                        except Exception as e:
                            logger.error(f"Error in second model verification: {e}")
                
                # Higher threshold for ambiguous matches
                required_confidence = min_confidence
                if multiple_matches:
                    required_confidence += 10  # Require higher confidence when there's ambiguity
                
                # Hard cutoff for very poor matches
                if distance > 0.5:  # Reduced from 0.55 for greater accuracy
                    logger.warning(f"Match rejected: Distance {distance:.4f} too high for {user_id}")
                    return False, (False, "Low Confidence", confidence)
                
                processing_time = time.time() - start_time
                logger.info(f"Found {user_id} with distance {distance:.4f} (confidence {confidence:.2f}%), took {processing_time:.3f}s")
                
                if confidence >= required_confidence:
                    with cache_lock:
                        recognition_cache[cache_key] = (True, user_id, confidence)
                        recognition_cache_timestamps[cache_key] = time.time()
                    
                    return True, user_id, confidence
                else:
                    logger.info(f"Match with {user_id} rejected, confidence {confidence:.2f}% below threshold {required_confidence}%")
                    return False, (False, "Low Confidence", confidence)
            
            # No match found
            processing_time = time.time() - start_time
            logger.warning(f"No match found, took {processing_time:.3f}s")
            return False, "Unknown", 0
                
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return False, "Error", 0
    
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return False, "Error", 0

#######################################################
# SECTION 3: IMAGE PROCESSING UTILITIES
#######################################################

def resize_if_large(image, max_size):
    """Resize image if either dimension exceeds max_size for faster processing"""
    if image is None:
        return None
        
    height, width = image.shape[:2]
    
    # Skip if already small enough
    if height <= max_size and width <= max_size:
        return image
        
    # Calculate the scaling factor
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def efficient_preprocess(image, target_size=(160, 160)):
    """Enhanced preprocessing for better recognition quality"""
    try:
        if image is None or image.size == 0:
            return None
        
        # Apply histogram equalization to improve contrast
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Split channels
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            # Merge enhanced L channel with original A and B channels
            enhanced_lab = cv2.merge((cl, a, b))
            # Convert back to BGR
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_image = clahe.apply(image)
        
        # Resize with improved quality
        resized = cv2.resize(enhanced_image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize pixel values
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply light Gaussian blur to reduce noise (helps with recognition)
        if target_size[0] >= 100:
            blurred = cv2.GaussianBlur(normalized, (3, 3), 0.5)
            return blurred
        
        return normalized
    except Exception as e:
        logger.error(f"Enhanced preprocessing error: {e}")
        return None

#######################################################
# SECTION 4: REFERENCE IMAGE MANAGEMENT
#######################################################

def load_reference_images():
    """Load reference images with subfolder organization by employee"""
    logger.info("Loading reference images...")
    
    reference_images = {}
    reference_names = []
    user_reference_images.clear()
    
    # Create directories if they don't exist
    os.makedirs('static/cache', exist_ok=True)
    os.makedirs('static/references', exist_ok=True)
    
    references_dir = 'static/references'
    if not os.path.exists(references_dir) or len(os.listdir(references_dir)) == 0:
        logger.warning("No reference images found")
        return {}, []
    
    # Get list of employee folders for parallel processing
    employee_folders = []
    for employee_folder in os.listdir(references_dir):
        employee_dir = os.path.join(references_dir, employee_folder)
        if os.path.isdir(employee_dir) and not employee_folder.startswith('.') and "_without" not in employee_folder:
            employee_folders.append((employee_folder, employee_dir))
    
    # Load employee folders in parallel if enabled
    if USE_THREADING and len(employee_folders) > 2:
        # Create a list to store futures
        futures = []
        for employee_id, folder_path in employee_folders:
            future = executor.submit(load_employee_folder, employee_id, folder_path)
            futures.append((employee_id, future))
        
        # Process results as they complete
        for employee_id, future in futures:
            try:
                employee_images = future.result()
                if employee_images:
                    user_reference_images[employee_id] = employee_images
                    reference_images[employee_id] = employee_images[0]  # First image as primary
                    reference_names.append(employee_id)
            except Exception as e:
                logger.error(f"Error loading employee {employee_id} in parallel: {e}")
    else:
        # Sequential loading for small number of employees
        for employee_id, folder_path in employee_folders:
            employee_images = load_employee_folder(employee_id, folder_path)
            if employee_images:
                user_reference_images[employee_id] = employee_images
                reference_images[employee_id] = employee_images[0]  
                reference_names.append(employee_id)
    
    total_references = sum(len(imgs) for imgs in user_reference_images.values())
    logger.info(f"Loaded {len(reference_images)} users with {total_references} references")
    
    return reference_images, reference_names

def load_employee_folder(employee_id, folder_path):
    """Load all reference images for an employee from their folder with quality filtering"""
    employee_images = []
    
    # Check if this is a main folder with subfolders
    with_glasses_path = os.path.join(folder_path, "with glass")
    without_glasses_path = os.path.join(folder_path, "without glass")
    
    # Check if we're using the old subfolder structure or the new direct structure
    old_structure = (os.path.exists(with_glasses_path) and os.path.isdir(with_glasses_path)) or \
                   (os.path.exists(without_glasses_path) and os.path.isdir(without_glasses_path))
    
    # Process all images in the main folder (new simplified structure)
    try:
        # Get all image files in the main folder
        image_files = [f for f in os.listdir(folder_path) if 
                      f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        # Group images by angle for strategic sampling
        angle_groups = {}
        
        for image_file in image_files:
            try:
                # Extract angle information from filename
                angle_info = None
                if "front" in image_file.lower():
                    angle_info = "front"
                elif "left" in image_file.lower():
                    angle_info = "left"
                elif "right" in image_file.lower():
                    angle_info = "right"
                elif "up" in image_file.lower():
                    angle_info = "up"
                elif "down" in image_file.lower():
                    angle_info = "down"
                else:
                    angle_info = "unknown"
                    
                # Add glasses info if the filename indicates glasses
                if "with_glasses" in image_file.lower():
                    angle_info += "_glasses"
                    
                # Group by angle
                if angle_info not in angle_groups:
                    angle_groups[angle_info] = []
                angle_groups[angle_info].append(image_file)
            except Exception as e:
                logger.error(f"Error processing filename {image_file}: {e}")
        
        # Process each angle group
        for angle, files in angle_groups.items():
            try:
                # Sort by timestamp (newest first) - assuming filenames have timestamps
                files.sort(reverse=True)
                
                # Limit number of references per angle to avoid processing too many
                max_per_angle = MAX_REFERENCES_PER_ANGLE
                
                # Process up to max_per_angle files for this angle
                for i, image_file in enumerate(files[:max_per_angle]):
                    try:
                        image_path = os.path.join(folder_path, image_file)
                        
                        # Skip reading problematic images
                        if "noface" in image_file.lower() or "quality" in image_file.lower():
                            logger.debug(f"Skipping low quality image: {image_file}")
                            continue
                        
                        # Read the image
                        img = cv2.imread(image_path)
                        if img is None:
                            logger.warning(f"Failed to read image: {image_path}")
                            continue
                            
                        # Validate the face
                        faces = detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.5)
                        if not faces:
                            logger.warning(f"No face detected in reference: {image_path}")
                            continue
                            
                        # Preprocess image for recognition
                        preprocessed = efficient_preprocess(img)
                        if preprocessed is not None:
                            employee_images.append(preprocessed)
                            
                    except Exception as e:
                        logger.error(f"Error processing reference image {image_file}: {e}")
            except Exception as e:
                logger.error(f"Error processing angle group {angle}: {e}")
                
    except Exception as e:
        logger.error(f"Error loading employee folder {folder_path}: {e}")
    
    # Handle old structure as a fallback if no images were loaded from main folder
    if old_structure and not employee_images:
        logger.info(f"Using old subfolder structure for {employee_id}")
        
        # Process main folder and any subfolders if they exist
        folders_to_process = [folder_path]  # Always process the main folder
        
        # Add subfolders if they exist
        if os.path.exists(with_glasses_path) and os.path.isdir(with_glasses_path):
            folders_to_process.append(with_glasses_path)
        if os.path.exists(without_glasses_path) and os.path.isdir(without_glasses_path):
            folders_to_process.append(without_glasses_path)
        
        # Process each folder
        for current_folder in folders_to_process:
            # Get all image files in this folder
            subfolder_files = [f for f in os.listdir(current_folder) if 
                      f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            # Process a limited number of files from each subfolder
            for image_file in subfolder_files[:MAX_REFERENCES_PER_ANGLE]:
                try:
                    image_path = os.path.join(current_folder, image_file)
                    
                    # Skip reading problematic images
                    if "noface" in image_file.lower() or "quality" in image_file.lower():
                        continue
                    
                    # Read the image
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                        
                    # Validate the face
                    faces = detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.5)
                    if not faces:
                        continue
                        
                    # Preprocess image for recognition
                    preprocessed = efficient_preprocess(img)
                    if preprocessed is not None:
                        employee_images.append(preprocessed)
                        
                except Exception as e:
                    logger.error(f"Error processing old structure image {image_file}: {e}")
    
    # Log the number of references loaded
    logger.info(f"Loaded {len(employee_images)} reference images for {employee_id}")
    
    return employee_images

def capture_multiple_angles(employee_id, angle_images):
    """Store multiple angle images for an employee with optimized reference count"""
    try:
        # Create main folder with employee ID
        references_dir = 'static/references'
        os.makedirs(references_dir, exist_ok=True)
        
        # Clean up any existing folders with similar employee_id
        # First delete any potential without_glasses folder
        old_without_folder = os.path.join(references_dir, f"{employee_id}_without_glass")
        if os.path.exists(old_without_folder):
            import shutil
            try:
                shutil.rmtree(old_without_folder)
                logger.info(f"Removed old folder {old_without_folder}")
            except Exception as e:
                logger.error(f"Error removing old folder {old_without_folder}: {e}")
                
        # Alternative naming
        old_without_folder2 = os.path.join(references_dir, f"{employee_id}_without_glasses")
        if os.path.exists(old_without_folder2):
            import shutil
            try:
                shutil.rmtree(old_without_folder2)
                logger.info(f"Removed old folder {old_without_folder2}")
            except Exception as e:
                logger.error(f"Error removing old folder {old_without_folder2}: {e}")
        
        # Create or ensure employee folder exists
        employee_folder = os.path.join(references_dir, employee_id)
        os.makedirs(employee_folder, exist_ok=True)
        logger.info(f"Using folder for employee: {employee_folder}")
        
        # Remove any existing subfolders to consolidate all photos in one place
        with_glass_subfolder = os.path.join(employee_folder, "with glass")
        without_glass_subfolder = os.path.join(employee_folder, "without glass")
        
        # Move any existing files from subfolders to main folder
        if os.path.exists(with_glass_subfolder) and os.path.isdir(with_glass_subfolder):
            for file in os.listdir(with_glass_subfolder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        import shutil
                        old_path = os.path.join(with_glass_subfolder, file)
                        new_path = os.path.join(employee_folder, file)
                        shutil.copy2(old_path, new_path)
                        logger.info(f"Moved {old_path} to {new_path}")
                    except Exception as e:
                        logger.error(f"Error moving file {file}: {e}")
            
            # Remove the subfolder after moving files
            try:
                import shutil
                shutil.rmtree(with_glass_subfolder)
                logger.info(f"Removed subfolder {with_glass_subfolder}")
            except Exception as e:
                logger.error(f"Error removing subfolder {with_glass_subfolder}: {e}")
        
        if os.path.exists(without_glass_subfolder) and os.path.isdir(without_glass_subfolder):
            for file in os.listdir(without_glass_subfolder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        import shutil
                        old_path = os.path.join(without_glass_subfolder, file)
                        new_path = os.path.join(employee_folder, file)
                        shutil.copy2(old_path, new_path)
                        logger.info(f"Moved {old_path} to {new_path}")
                    except Exception as e:
                        logger.error(f"Error moving file {file}: {e}")
            
            # Remove the subfolder after moving files
            try:
                import shutil
                shutil.rmtree(without_glass_subfolder)
                logger.info(f"Removed subfolder {without_glass_subfolder}")
            except Exception as e:
                logger.error(f"Error removing subfolder {without_glass_subfolder}: {e}")
        
        # Process each angle image
        success_count = 0
        
        # Keep track of angles to manage reference count limits
        angle_counts = {}
        
        for angle_name, image_data in angle_images.items():
            try:
                # Skip empty images
                if not image_data or len(image_data) < 100:
                    continue
                    
                # Decode base64 image
                img_data = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.error(f"Failed to decode image for {angle_name}")
                    continue
                
                # Check for face and image quality
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detect_faces(img_rgb, 0.5)
                
                # Determine if this is a glasses variation
                has_glasses = "_glasses" in angle_name
                
                # Set appropriate file prefix based on face detection
                if not faces:
                    logger.warning(f"No face detected in {angle_name} image")
                    file_prefix = "noface_"
                else:
                    # Measure quality
                    largest_face = max(faces, key=lambda face: face[0][2] * face[0][3])
                    confidence = largest_face[1]
                    x, y, w, h = largest_face[0]
                    face_size_percent = (w * h) / (img.shape[0] * img.shape[1])

                    # Blurriness check (Laplacian variance)
                    face_region = img_rgb[y:y+h, x:x+w]
                    gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                    variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    if variance < 50:
                        logger.warning(f"Blurry face detected in {angle_name}: variance={variance:.2f}, skipping image.")
                        continue
                    
                    # Skip low-quality faces
                    if confidence < QUALITY_THRESHOLD or face_size_percent < 0.05:
                        logger.warning(f"Low quality face in {angle_name}: conf={confidence:.2f}, size={face_size_percent:.2f}")
                        file_prefix = f"quality{int(confidence*100)}_"
                    else:
                        file_prefix = ""
                
                # Extract angle information
                base_angle = angle_name.replace("_glasses", "")
                
                # Track angle count for limiting references
                angle_key = f"{base_angle}_{'glasses' if has_glasses else 'no_glasses'}"
                if angle_key not in angle_counts:
                    angle_counts[angle_key] = 0
                    
                # Check if we've exceeded maximum for this angle
                if angle_counts[angle_key] >= MAX_REFERENCES_PER_ANGLE:
                    logger.info(f"Skipping {angle_name} image - already have {MAX_REFERENCES_PER_ANGLE} for this angle")
                    continue
                    
                angle_counts[angle_key] += 1
                
                # Add timestamp for unique filenames
                timestamp = int(time.time())
                
                # Create the filename - include glasses info in the filename itself
                glasses_info = "_with_glasses" if has_glasses else ""
                filename = f"{file_prefix}{base_angle}{glasses_info}_{timestamp}.jpg"
                
                # Save directly to employee folder
                file_path = os.path.join(employee_folder, filename)
                
                # Save the image
                cv2.imwrite(file_path, img)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error saving {angle_name} image: {e}")
        
        # Check if we have at least one success
        if success_count == 0:
            logger.error("Failed to save any images")
            return False, "Failed to save any reference images"
            
        # Update loaded references
        load_reference_images()
        
        return True, success_count
        
    except Exception as e:
        logger.error(f"Error in capture_multiple_angles: {e}")
        return False, str(e)

def create_limited_variations(face_img):
    """Create enhanced variations for better recognition in different conditions"""
    variations = []
    
    try:
        # Convert to PIL for easier image adjustments
        pil_img = Image.fromarray(face_img)
        
        # 1. Brightness variations (more variations for realistic lighting conditions)
        brightness_factors = [0.7, 0.85, 1.15, 1.3]
        for factor in brightness_factors:
            try:
                enhancer = ImageEnhance.Brightness(pil_img)
                enhanced_img = enhancer.enhance(factor)
                np_img = np.array(enhanced_img)
                variations.append(np_img)
            except Exception:
                continue
        
        # 2. Contrast variations
        contrast_factors = [0.8, 1.2]
        for factor in contrast_factors:
            try:
                enhancer = ImageEnhance.Contrast(pil_img)
                enhanced_img = enhancer.enhance(factor)
                np_img = np.array(enhanced_img)
                variations.append(np_img)
            except Exception:
                continue
        
        # 3. Slight rotations for angle variations
        rotation_angles = [-7, -3, 3, 7]
        for angle in rotation_angles:
            try:
                rotated_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
                np_img = np.array(rotated_img)
                variations.append(np_img)
            except Exception:
                continue
        
        # 4. Add slight perspective transformations to simulate different camera angles
        height, width = face_img.shape[:2]
        for x_shift in [-0.05, 0.05]:
            for y_shift in [-0.05, 0.05]:
                try:
                    # Create perspective transform matrix
                    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                    dst_points = np.float32([
                        [width * abs(x_shift) if x_shift < 0 else 0, height * abs(y_shift) if y_shift < 0 else 0],
                        [width * (1 - abs(x_shift)) if x_shift > 0 else width, height * abs(y_shift) if y_shift < 0 else 0],
                        [width * (1 - abs(x_shift)) if x_shift > 0 else width, height * (1 - abs(y_shift)) if y_shift > 0 else height],
                        [width * abs(x_shift) if x_shift < 0 else 0, height * (1 - abs(y_shift)) if y_shift > 0 else height]
                    ])
                    
                    # Apply perspective transformation
                    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    warped_img = cv2.warpPerspective(face_img, transform_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
                    variations.append(warped_img)
                except Exception:
                    continue
                
        # 5. Add slight color temperature adjustments to simulate different lighting conditions
        b, g, r = cv2.split(face_img)
        
        # Warmer (more orange/yellow)
        b_warm = np.clip(b * 0.9, 0, 255).astype(np.uint8)
        r_warm = np.clip(r * 1.1, 0, 255).astype(np.uint8)
        warm_img = cv2.merge([b_warm, g, r_warm])
        variations.append(warm_img)
        
        # Cooler (more blue)
        b_cool = np.clip(b * 1.1, 0, 255).astype(np.uint8)
        r_cool = np.clip(r * 0.9, 0, 255).astype(np.uint8)
        cool_img = cv2.merge([b_cool, g, r_cool])
        variations.append(cool_img)
        
        # Ensure we don't return too many variations to avoid memory issues
        if len(variations) > 12:
            # Return a good mix of variations by taking samples from start, middle, and end
            indices = np.linspace(0, len(variations)-1, 12).astype(int)
            variations = [variations[i] for i in indices]
        
        # Preprocess all variations for consistency
        processed_variations = []
        for var in variations:
            processed = efficient_preprocess(var)
            if processed is not None:
                processed_variations.append(processed)
        
        return processed_variations
        
    except Exception as e:
        logger.error(f"Error creating variations: {e}")
        return [efficient_preprocess(face_img)] if face_img is not None else []

# Optimized function to generate reference images with better speed
def capture_multiple_references(employee_id, base_image_data):
    """Generate multiple reference images in employee subfolder with optimized speed"""
    try:
        start_time = time.time()
        
        # Decode the base64 image
        img_data = base64.b64decode(base_image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, 0
        
        # Resize large images for faster processing
        img = resize_if_large(img, MAX_IMAGE_SIZE)
        
        # Check for face
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(img_rgb, 0.7)  # High confidence for reference images
        
        if not faces:
            logger.warning("No face detected in reference image")
            return False, 0
        
        # Get face region for transformations
        x, y, w, h = faces[0][0]
        face_img = img_rgb[y:y+h, x:x+w]
        
        # Create parent directory
        references_dir = 'static/references'
        os.makedirs(references_dir, exist_ok=True)
        
        # Create employee subfolder
        employee_dir = os.path.join(references_dir, employee_id)
        os.makedirs(employee_dir, exist_ok=True)
        
        # Save original image in employee subfolder
        orig_filename = os.path.join(employee_dir, "original.jpg")
        cv2.imwrite(orig_filename, img)
        
        # Create augmented versions with more variety - optimized for parallel processing
        augmentations = []
        
        # Define all augmentations we want to perform
        # Brightness variations
        augmentations.append(('brightness', 0.8))
        augmentations.append(('brightness', 1.2))
        
        # Rotation variations
        augmentations.append(('rotation', -5))
        augmentations.append(('rotation', 5))
        augmentations.append(('rotation', -10))
        augmentations.append(('rotation', 10))
        
        # Horizontal flip
        augmentations.append(('flip', 1))
        
        # Zoom variations
        augmentations.append(('zoom', 0.9))
        augmentations.append(('zoom', 1.1))
        
        # Use parallel processing to generate augmentations if enabled
        augmented_images = []
        
        if USE_THREADING and len(augmentations) > 4:
            # Function to perform a single augmentation
            def perform_augmentation(aug_type, param):
                if aug_type == 'brightness':
                    return cv2.convertScaleAbs(face_img, alpha=param, beta=0)
                elif aug_type == 'rotation':
                    rows, cols = face_img.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), param, 1)
                    return cv2.warpAffine(face_img, M, (cols, rows))
                elif aug_type == 'flip':
                    return cv2.flip(face_img, param)
                elif aug_type == 'zoom':
                    h, w = face_img.shape[:2]
                    center_x, center_y = w//2, h//2
                    new_w, new_h = int(w*param), int(h*param)
                    left = max(0, center_x - new_w//2)
                    top = max(0, center_y - new_h//2)
                    right = min(w, center_x + new_w//2)
                    bottom = min(h, center_y + new_h//2)
                    cropped = face_img[top:bottom, left:right]
                    return cv2.resize(cropped, (w, h))
                return None
            
            # Submit all augmentation tasks to thread pool
            futures = []
            for aug_type, param in augmentations:
                future = executor.submit(perform_augmentation, aug_type, param)
                futures.append(future)
            
            # Collect results
            for future in futures:
                aug_img = future.result()
                if aug_img is not None:
                    augmented_images.append(aug_img)
        else:
            # Sequential processing for small number of augmentations
            for aug_type, param in augmentations:
                if aug_type == 'brightness':
                    bright_img = cv2.convertScaleAbs(face_img, alpha=param, beta=0)
                    augmented_images.append(bright_img)
                elif aug_type == 'rotation':
                    rows, cols = face_img.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), param, 1)
                    rotated = cv2.warpAffine(face_img, M, (cols, rows))
                    augmented_images.append(rotated)
                elif aug_type == 'flip':
                    flipped = cv2.flip(face_img, param)
                    augmented_images.append(flipped)
                elif aug_type == 'zoom':
                    h, w = face_img.shape[:2]
                    center_x, center_y = w//2, h//2
                    new_w, new_h = int(w*param), int(h*param)
                    left = max(0, center_x - new_w//2)
                    top = max(0, center_y - new_h//2)
                    right = min(w, center_x + new_w//2)
                    bottom = min(h, center_y + new_h//2)
                    cropped = face_img[top:bottom, left:right]
                    zoomed = cv2.resize(cropped, (w, h))
                    augmented_images.append(zoomed)
        
        # Save each augmented image in employee subfolder
        for i, aug_img in enumerate(augmented_images):
            aug_filename = os.path.join(employee_dir, f"augmented_{i+1:03d}.jpg")
            cv2.imwrite(aug_filename, aug_img)
        
        # Clear caches and force reload
        with cache_lock:
            recognition_cache.clear()
            recognition_cache_timestamps.clear()
            
        processing_time = time.time() - start_time
        logger.info(f"Generated {len(augmented_images)+1} reference images for {employee_id} in {processing_time:.3f}s")
        return True, len(augmented_images) + 1
    
    except Exception as e:
        logger.error(f"Error creating references: {e}")
        return False, 0

# Simplified cache cleaning function
def clean_cache_periodically():
    """Clean the recognition cache periodically"""
    while True:
        time.sleep(300)  # Run every 5 minutes
        try:
            with cache_lock:
                current_time = time.time()
                expired_keys = [k for k, v in recognition_cache_timestamps.items() 
                              if current_time - v > CACHE_DURATION]
                
                for key in expired_keys:
                    recognition_cache.pop(key, None)
                    recognition_cache_timestamps.pop(key, None)
                
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")

# Start the cache cleaning thread
cache_cleaner = threading.Thread(target=clean_cache_periodically, daemon=True)
cache_cleaner.start()

def delete_user(employee_id):
    """Delete all reference images and folders for a given employee from static/references."""
    import shutil
    references_dir = 'static/references'
    deleted_any = False
    
    # Main employee folder
    employee_folder = os.path.join(references_dir, employee_id)
    if os.path.exists(employee_folder) and os.path.isdir(employee_folder):
        try:
            shutil.rmtree(employee_folder)
            logger.info(f"Deleted reference folder for employee: {employee_id}")
            deleted_any = True
        except Exception as e:
            logger.error(f"Error deleting folder {employee_folder}: {e}")
    
    # Old folder structures
    old_without_folder = os.path.join(references_dir, f"{employee_id}_without_glass")
    if os.path.exists(old_without_folder) and os.path.isdir(old_without_folder):
        try:
            shutil.rmtree(old_without_folder)
            logger.info(f"Deleted old folder: {old_without_folder}")
            deleted_any = True
        except Exception as e:
            logger.error(f"Error deleting old folder {old_without_folder}: {e}")
    old_without_folder2 = os.path.join(references_dir, f"{employee_id}_without_glasses")
    if os.path.exists(old_without_folder2) and os.path.isdir(old_without_folder2):
        try:
            shutil.rmtree(old_without_folder2)
            logger.info(f"Deleted old folder: {old_without_folder2}")
            deleted_any = True
        except Exception as e:
            logger.error(f"Error deleting old folder {old_without_folder2}: {e}")
    
    # Optionally, refresh loaded references
    load_reference_images()
    
    return deleted_any