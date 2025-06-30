// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const faceGuide = document.getElementById('face-guide');
const processingIndicator = document.getElementById('processing-indicator');
const matchText = document.getElementById('match-text');
const recognitionStatus = document.getElementById('recognitionStatus');
const recognizedName = document.getElementById('recognizedName');
const confidenceContainer = document.getElementById('confidence-container');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const errorMessage = document.getElementById('error-message');
const toggleBtn = document.getElementById('toggleBtn');
const scanningProgressContainer = document.getElementById('scanning-progress-container');
const scanningProgressBar = document.getElementById('scanning-progress-bar');

// Global variables
let stream = null;
let recognizing = false;
let isActive = true;
let faceDetectionTimer = null;
let resultDisplayTimer = null;
let lastRecognizedId = null;
let lastRecognitionTime = 0;
let detectionCooldown = false;
let isScanningPaused = false;
let faceCheckTimer = null;
let progressAnimationId = null;
let consecutiveFaceDetections = 0;
let consecutiveNoFaceDetections = 0;

// Face detection stability buffer
let faceDetectionBuffer = [];
let faceDetectionBufferSize = 6; // Number of frames to maintain for stability
let faceDetectionThreshold = 0.6; // 60% positive detections to consider a face present
let lastDetectionState = false; // Track last stable detection state

// Start the camera
function startCamera() {
    if (stream) return; // Camera already started
    
    recognitionStatus.textContent = 'Starting camera...';
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            width: 640, 
            height: 480,
            facingMode: 'user'
        }
    })
    .then(function(videoStream) {
        stream = videoStream;
        video.srcObject = stream;
        
        video.onloadedmetadata = function() {
            recognitionStatus.textContent = 'Camera ready. Waiting for faces...';
            matchText.textContent = 'WAITING FOR FACE';
            
            // Wait for camera to stabilize
            setTimeout(startFaceMonitoring, 2000);
        };
    })
    .catch(function(error) {
        console.error('Camera error:', error);
        recognitionStatus.textContent = 'Camera error: ' + error.message;
        errorMessage.textContent = 'Failed to access camera. Please check your camera and permissions.';
        errorMessage.style.display = 'block';
    });
    // Add this at the end of the startCamera function
video.onloadedmetadata = function() {
    recognitionStatus.textContent = 'Camera ready. Waiting for faces...';
    matchText.textContent = 'WAITING FOR FACE';
    
    // Wait for camera to stabilize
    setTimeout(() => {
        startFaceMonitoring();
        
        // Force an immediate face check
        setTimeout(() => {
            if (isActive) {
                // Do multiple checks with short intervals to ensure a face is detected
                checkForFacePresence();
                setTimeout(checkForFacePresence, 300);
                setTimeout(checkForFacePresence, 600);
                setTimeout(checkForFacePresence, 900);
            }
        }, 1000);
    }, 2000);
};
}

// Initialize camera on page load
window.addEventListener('DOMContentLoaded', startCamera);

// Function to record successful or denied access
function recordAccess(data) {
    fetch('/record_access', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .catch(error => {
        console.error('Error recording access:', error);
    });
}

// Function to record failed access
function recordAccessFailure() {
    fetch('/record_access', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            match: false,
            id: 'Unknown',
            name: 'Unknown',
            status: 'N/A',
            access: 'Denied'
        })
    })
    .catch(error => {
        console.error('Error recording access failure:', error);
    });
}

// Toggle button handler
toggleBtn.addEventListener('click', toggleRecognition);

// Clean up when leaving the page
window.addEventListener('beforeunload', function() {
    if (faceDetectionTimer) {
        clearInterval(faceDetectionTimer);
    }
    
    if (faceCheckTimer) {
        clearInterval(faceCheckTimer);
    }
    
    if (resultDisplayTimer) {
        clearTimeout(resultDisplayTimer);
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});

// Start monitoring for faces (lighter process than full detection)
function startFaceMonitoring() {
    if (!isActive) return;
    
    // Clear any existing timers
    if (faceCheckTimer) {
        clearInterval(faceCheckTimer);
    }
    
    // Reset counters
    consecutiveFaceDetections = 0;
    consecutiveNoFaceDetections = 0;
    
    // Check more frequently for responsive detection
    faceCheckTimer = setInterval(() => {
        if (!isActive || recognizing || detectionCooldown) return;
        
        // Check if there's a face
        checkForFacePresence();
    }, 150); // Check every 150ms for better responsiveness
    
    recognitionStatus.textContent = 'Waiting for a face to appear...';
    matchText.textContent = 'WAITING FOR FACE';
    matchText.className = 'match-text';
    isScanningPaused = true;
    
    // Do an immediate check to detect any face already in the frame
    setTimeout(checkForFacePresence, 100);
}

// Improved face detection using a robust algorithm with better lighting checks
function hasFace(imageData) {
    // Create a temporary canvas to process the image
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // Set dimensions
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    
    // Draw the image data onto the canvas
    tempCtx.putImageData(imageData, 0, 0);
    
    // Convert to grayscale for better processing
    const grayscaleData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = grayscaleData.data;
    
    let totalBrightness = 0; // This variable was missing
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114);
        data[i] = data[i+1] = data[i+2] = avg;
        totalBrightness += avg;
    }
    
    // Convert to grayscale using standard formula
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114);
        data[i] = data[i+1] = data[i+2] = avg;
    }
    
    // Calculate histogram to detect blank or low-contrast images
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
        histogram[data[i]]++;
    }
    
    // Calculate histogram standard deviation - low std dev means uniform image (no face)
    let mean = 0;
    for (let i = 0; i < 256; i++) {
        mean += i * histogram[i];
    }
    mean /= (data.length / 4);
    
    let variance = 0;
    for (let i = 0; i < 256; i++) {
        variance += histogram[i] * (i - mean) * (i - mean);
    }
    variance /= (data.length / 4);
    const stdDev = Math.sqrt(variance);
    
    // If image has very low contrast, it's likely blank or a wall - not a face
    if (stdDev < 20) {
        return false;
    }
    
    // Focus on the central region where the face is likely to be
    const centerX = Math.floor(tempCanvas.width / 2);
    const centerY = Math.floor(tempCanvas.height / 2);
    const faceRegionWidth = Math.floor(tempCanvas.width * 0.5); // Tighter focus region
    const faceRegionHeight = Math.floor(tempCanvas.height * 0.6); // Tighter focus region
    const startX = centerX - Math.floor(faceRegionWidth / 2);
    const startY = centerY - Math.floor(faceRegionHeight / 2);
    const endX = startX + faceRegionWidth;
    const endY = startY + faceRegionHeight;
    
    // Apply edge detection with higher threshold
    let edgeCount = 0;
    const edgeThreshold = 35; // Increased threshold - more strict
    
    // Count edges only in the face region
    for (let y = startY + 1; y < endY - 1; y += 2) {
        for (let x = startX + 1; x < endX - 1; x += 2) {
            const idx = (y * tempCanvas.width + x) * 4;
            
            // Simplified Sobel operator for edge detection
            const top = data[((y-1) * tempCanvas.width + x) * 4];
            const bottom = data[((y+1) * tempCanvas.width + x) * 4];
            const left = data[(y * tempCanvas.width + (x-1)) * 4];
            const right = data[(y * tempCanvas.width + (x+1)) * 4];
            
            const verticalGradient = Math.abs(top - bottom);
            const horizontalGradient = Math.abs(left - right);
            const gradient = verticalGradient + horizontalGradient;
            
            if (gradient > edgeThreshold) {
                edgeCount++;
            }
        }
    }
    
    // Calculate edge density in the face region
    const totalPixels = (faceRegionWidth * faceRegionHeight) / 4; // Adjusted for skipping pixels
    const edgeDensity = (edgeCount / totalPixels);
    
    // Reasonable brightness range
    const avgBrightness = totalBrightness / (data.length / 4);
    const isReasonableBrightness = avgBrightness > 50 && avgBrightness < 200;
    
    // Stricter edge density threshold
    const faceThreshold = 0.025; // Increased from 0.018
    
    // Higher minimum edge count
    const minEdgeCount = 150; // Increased from 100
    
    // Check edge distribution - faces should have edges throughout
    // Divide the face region into 4 quadrants and check each one has edges
    let quadrantEdges = [0, 0, 0, 0];
    const halfWidth = Math.floor(faceRegionWidth / 2);
    const halfHeight = Math.floor(faceRegionHeight / 2);
    
    for (let y = startY + 1; y < endY - 1; y += 2) {
        for (let x = startX + 1; x < endX - 1; x += 2) {
            const idx = (y * tempCanvas.width + x) * 4;
            
            const top = data[((y-1) * tempCanvas.width + x) * 4];
            const bottom = data[((y+1) * tempCanvas.width + x) * 4];
            const left = data[(y * tempCanvas.width + (x-1)) * 4];
            const right = data[(y * tempCanvas.width + (x+1)) * 4];
            
            const gradient = Math.abs(top - bottom) + Math.abs(left - right);
            
            if (gradient > edgeThreshold) {
                // Determine which quadrant this pixel is in
                const quadX = x < (startX + halfWidth) ? 0 : 1;
                const quadY = y < (startY + halfHeight) ? 0 : 1;
                const quadrant = quadY * 2 + quadX;
                quadrantEdges[quadrant]++;
            }
        }
    }
    
    // Require a minimum number of edges in each quadrant
    const minEdgesPerQuadrant = 20;
    const hasGoodDistribution = quadrantEdges.every(count => count >= minEdgesPerQuadrant);
    
    // Face detection final decision - all criteria must be met
    const result = edgeDensity > faceThreshold && 
                  isReasonableBrightness && 
                  edgeCount > minEdgeCount &&
                  hasGoodDistribution;
    
    // Debug output - can be removed in production
    console.log(`StdDev: ${stdDev.toFixed(1)}, Brightness: ${avgBrightness.toFixed(1)}, Edges: ${edgeCount}, Density: ${edgeDensity.toFixed(4)}, Distribution: ${hasGoodDistribution}, Face: ${result}`);
    
    return result;
}

// Improved face presence check with stability buffer
// Modify this function in recognition.js
function checkForFacePresence() {
    if (recognizing || !isActive || detectionCooldown) return;
    
    // Capture current frame
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    
    // Look for skin-tone colors in the center of the frame
    const centerX = Math.floor(canvas.width / 2);
    const centerY = Math.floor(canvas.height / 2);
    const sampleRadius = Math.min(canvas.width, canvas.height) * 0.25;
    
    let skinTonePixels = 0;
    let totalPixels = 0;
    
    for (let y = centerY - sampleRadius; y < centerY + sampleRadius; y += 2) {
        for (let x = centerX - sampleRadius; x < centerX + sampleRadius; x += 2) {
            if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
                const idx = (y * canvas.width + x) * 4;
                const r = imageData.data[idx];
                const g = imageData.data[idx + 1];
                const b = imageData.data[idx + 2];
                
                // Simple skin tone detection
                // Skin has more red than blue, and red and green are moderately high
                if (r > 60 && g > 40 && b > 20 && r > b && (r-b) > 15) {
                    skinTonePixels++;
                }
                
                totalPixels++;
            }
        }
    }
    
    // Calculate percentage of skin tone pixels
    const skinTonePercentage = (skinTonePixels / totalPixels) * 100;
    
    // Debug output
    console.log(`Skin tone pixels: ${skinTonePercentage.toFixed(1)}%`);
    
    // Use consecutive detection counters for more stable transitions
    if (skinTonePercentage > 15) { // Lower threshold to be more sensitive
        consecutiveFaceDetections++;
        consecutiveNoFaceDetections = 0;
    } else {
        consecutiveNoFaceDetections++;
        consecutiveFaceDetections = 0;
    }
    
    // State transitions based on consecutive detections
    if (isScanningPaused && consecutiveFaceDetections >= 3) {
        // Start recognition when we have 3 consecutive face detections
        console.log("Face detected - starting recognition!");
        isScanningPaused = false;
        startFaceDetection();
    } 
    else if (!isScanningPaused && !recognizing && !detectionCooldown && 
             consecutiveNoFaceDetections >= 5) {
        // Stop recognition when we have 5 consecutive no-face detections
        console.log("Face not detected - pausing recognition");
        isScanningPaused = true;
        if (faceDetectionTimer) {
            clearInterval(faceDetectionTimer);
            faceDetectionTimer = null;
        }
        
        matchText.textContent = 'WAITING FOR FACE';
        matchText.className = 'match-text';
        recognitionStatus.textContent = 'No face detected. Waiting for a face to appear...';
        
        // Clear any displayed results
        recognizedName.textContent = '';
        confidenceContainer.style.display = 'none';
    }
}
// Start active face detection and recognition
function startFaceDetection() {
    if (!isActive) return;
    
    // Clear any existing timers
    if (faceDetectionTimer) {
        clearInterval(faceDetectionTimer);
    }
    
    // Set up continuous face detection with a longer interval
    faceDetectionTimer = setInterval(() => {
        if (!isActive || recognizing || detectionCooldown || isScanningPaused) return;
        
        // First, verify there is still a face present before processing
        const context = canvas.getContext('2d');
        canvas.width = 320;
        canvas.height = 240;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        
        if (hasFace(imageData)) {
            // Face is still present, proceed with recognition
            doFaceRecognition();
        } else {
            // Don't immediately switch to monitoring mode - let the buffer system handle this
            // This prevents premature abandonment of recognition mode due to a single frame
            // without detection
        }
    }, 1500); // Increased from 800ms to 1500ms for better synchronization
    
    recognitionStatus.textContent = 'Face detected! Starting recognition...';
    matchText.textContent = 'SCANNING';
    matchText.className = 'match-text';
}

// Function to start progress animation
function startProgressAnimation() {
    if (progressAnimationId) return; // Already running
    
    // Show the progress container
    scanningProgressContainer.style.display = 'block';
    
    // Reset progress
    let progress = 0;
    scanningProgressBar.style.width = '0%';
    
    // Animate the progress bar with a slower, more consistent pace
    progressAnimationId = setInterval(() => {
        // Adjust increment to be slower
        if (progress < 70) {
            // Move faster at the beginning
            progress += 2; // Reduced from 3 to 2
        } else if (progress < 90) {
            // Medium pace in the middle
            progress += 1;
        } else {
            // Very slow at the end
            progress += 0.3; // Reduced from 0.5 to 0.3
        }
        
        // Cap at 99% - the final 100% happens when process completes
        if (progress > 99) progress = 99;
        
        // Update the bar
        scanningProgressBar.style.width = `${progress}%`;
    }, 70); // Increased from 50ms to 70ms to make animation slower
}

// Function to stop progress animation and finalize
function stopProgressAnimation(success = true) {
    // Clear the interval
    if (progressAnimationId) {
        clearInterval(progressAnimationId);
        progressAnimationId = null;
    }
    
    // Complete the progress bar
    scanningProgressBar.style.width = '100%';
    
    // If it was a failure, make the bar red
    if (!success) {
        scanningProgressBar.style.backgroundColor = '#e74c3c';
    } else {
        scanningProgressBar.style.backgroundColor = '#27ae60';
    }
    
    // Hide the progress bar after a delay
    setTimeout(() => {
        scanningProgressContainer.style.display = 'none';
        // Reset the color for next time
        scanningProgressBar.style.backgroundColor = '#27ae60';
    }, 1000);
}

// Toggle recognition active/paused
function toggleRecognition() {
    isActive = !isActive;
    
    if (isActive) {
        // Reset state variables when restarting recognition
        recognizing = false;
        detectionCooldown = false;
        lastRecognizedId = null;
        faceDetectionBuffer = [];
        lastDetectionState = false;
        
        // Clear any existing timers
        if (faceDetectionTimer) {
            clearInterval(faceDetectionTimer);
            faceDetectionTimer = null;
        }
        
        if (faceCheckTimer) {
            clearInterval(faceCheckTimer);
            faceCheckTimer = null;
        }
        
        if (resultDisplayTimer) {
            clearTimeout(resultDisplayTimer);
            resultDisplayTimer = null;
        }
        
        toggleBtn.textContent = 'Stop Recognition';
        recognitionStatus.textContent = 'Waiting for faces...';
        matchText.textContent = 'WAITING FOR FACE';
        matchText.className = 'match-text';
        
        // Reset UI elements
        recognizedName.textContent = '';
        confidenceContainer.style.display = 'none';
        
        // Force an immediate face check instead of waiting for the timer
        isScanningPaused = true;
        startFaceMonitoring();
        
        // Force an immediate check for faces
        setTimeout(() => {
            if (isActive) {
                checkForFacePresence();
            }
        }, 500);
    } else {
        toggleBtn.textContent = 'Start Recognition';
        recognitionStatus.textContent = 'Recognition stopped. Click Start to continue.';
        matchText.textContent = 'STOPPED';
        matchText.className = 'match-text no-match';
        
        if (faceDetectionTimer) {
            clearInterval(faceDetectionTimer);
            faceDetectionTimer = null;
        }
        
        if (faceCheckTimer) {
            clearInterval(faceCheckTimer);
            faceCheckTimer = null;
        }
        
        // Clear any display timers
        if (resultDisplayTimer) {
            clearTimeout(resultDisplayTimer);
            resultDisplayTimer = null;
        }
    }
}

// Find this section in recognition.js
function updateConfidenceDisplay(confidence) {
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceValue');
    
    // Update the confidence bar
    confidenceBar.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence.toFixed(1)}%`;
    
    // Add color coding based on confidence
    if (confidence >= 85) {
        confidenceBar.style.backgroundColor = '#27ae60'; // Strong green
    } else if (confidence >= 70) {
        confidenceBar.style.backgroundColor = '#2ecc71'; // Green
    } else if (confidence >= 60) {
        confidenceBar.style.backgroundColor = '#f39c12'; // Orange
    } else {
        confidenceBar.style.backgroundColor = '#e74c3c'; // Red
    }
}

// Perform the face recognition
function doFaceRecognition() {
    if (recognizing || !isActive || isScanningPaused) return;
    recognizing = true;
    
    // Set a longer cooldown to prevent over-processing
    detectionCooldown = true;
    
    try {
        // Show processing indicator
        processingIndicator.style.display = 'block';
        
        // Start the progress animation
        startProgressAnimation();
        
        // Capture the current video frame
        const context = canvas.getContext('2d');
        
        // Use higher resolution for recognition
        const captureWidth = 400;  
        const captureHeight = 300;
        
        canvas.width = captureWidth;
        canvas.height = captureHeight;
        context.drawImage(video, 0, 0, captureWidth, captureHeight);
        
        // Double-check that a face is actually in the frame before sending
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        if (!hasFace(imageData)) {
            // No face detected in the final capture, abort processing
            processingIndicator.style.display = 'none';
            
            // Stop progress animation with failure
            stopProgressAnimation(false);
            
            recognizing = false;
            
            console.log("No face in final capture - aborting recognition");
            
            // Go back to monitoring mode after a delay
            setTimeout(() => {
                detectionCooldown = false;
                
                // Reset detection buffer
                faceDetectionBuffer = [];
                
                // Restart face monitoring
                isScanningPaused = true;
                if (faceDetectionTimer) {
                    clearInterval(faceDetectionTimer);
                    faceDetectionTimer = null;
                }
                
                matchText.textContent = 'WAITING FOR FACE';
                matchText.className = 'match-text';
                recognitionStatus.textContent = 'Face disappeared. Waiting for a face to appear...';
            }, 1000);
            
            return;
        }
        
        // Use medium quality JPEG for faster upload but maintain details
        const imageDataUrl = canvas.toDataURL('image/jpeg', 0.85);
        
        // Update UI
        matchText.textContent = 'PROCESSING';
        recognitionStatus.textContent = 'Processing face...';
        
        // Send to server for recognition
        fetch('/process_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageDataUrl })
        })
        .then(response => {
            if (response.status === 429) {
                // Stop progress animation with failure
                stopProgressAnimation(false);
                throw new Error('Too many requests. Please wait a moment and try again.');
            }
            return response.json();
        })
        .then(data => {
            // Hide processing indicator
            processingIndicator.style.display = 'none';
            
            // Complete the progress animation with success
            stopProgressAnimation(true);
            
            // Handle recognition results
            if (data.match) {
                // Check if this is the same person recognized very recently (within 5 seconds)
                const currentTime = Date.now();
                const timeSinceLastRecognition = currentTime - lastRecognitionTime;
                
                if (data.id === lastRecognizedId && timeSinceLastRecognition < 5000) {
                    // Same person recognized again very quickly, skip showing result
                    // Release the detection lock after a short delay
                    setTimeout(() => {
                        recognizing = false;
                        // Keep cooldown active for longer to prevent too frequent recognitions
                        setTimeout(() => {
                            detectionCooldown = false;
                        }, 1500);
                    }, 800);
                    
                    return;
                }
                
                // Update recognition time and ID
                lastRecognizedId = data.id;
                lastRecognitionTime = currentTime;
                
                // Update UI with match result
                recognitionStatus.textContent = 'Match found!';
                recognizedName.textContent = `Name: ${data.name}`;
                
                // Show confidence bar
                confidenceContainer.style.display = 'block';
                confidenceBar.style.width = `${data.confidence}%`;
                confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;
                
                if (data.access === 'Granted') {
                    matchText.textContent = 'ACCESS GRANTED';
                    matchText.className = 'match-text match';
                } else {
                    matchText.textContent = 'ACCESS DENIED';
                    matchText.className = 'match-text no-match';
                }
                
                // Record the access only if not already recorded by server
                if (!data.recorded) {
                    recordAccess(data);
                }
                
                // Display result for 3 seconds before resuming scanning
                resultDisplayTimer = setTimeout(() => {
                    // Release the detection lock
                    recognizing = false;
                    
                    // Clean up the display after showing result
                    if (isActive) {
                        // Check if face is still present before resetting
                        const checkContext = canvas.getContext('2d');
                        canvas.width = 320;
                        canvas.height = 240;
                        checkContext.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const checkImageData = checkContext.getImageData(0, 0, canvas.width, canvas.height);
                        
                        // If face is still present, continue scanning instead of waiting
                        if (hasFace(checkImageData)) {
                            console.log("Face still present after result - continuing scanning");
                            // Reset detection buffer but keep scanning active
                            faceDetectionBuffer = [];
                            isScanningPaused = false;
                            
                            // Update UI to indicate continued scanning
                            matchText.textContent = 'SCANNING';
                            matchText.className = 'match-text';
                            recognitionStatus.textContent = 'Continuing to scan...';
                            recognizedName.textContent = '';
                            confidenceContainer.style.display = 'none';
                            
                            // Restart active detection immediately
                            if (!faceDetectionTimer) {
                                startFaceDetection();
                            }
                            
                            // Release cooldown after a longer delay to prevent too frequent recognitions
                            setTimeout(() => {
                                detectionCooldown = false;
                            }, 2500); // Increased from 500ms to 2500ms
                        } else {
                            // No face detected, go to waiting mode
                            // Reset detection buffer for a fresh start
                            faceDetectionBuffer = [];
                            
                            // Reset scanning state
                            isScanningPaused = true;
                            
                            if (faceDetectionTimer) {
                                clearInterval(faceDetectionTimer);
                                faceDetectionTimer = null;
                            }
                            
                            matchText.textContent = 'WAITING FOR FACE';
                            matchText.className = 'match-text';
                            recognitionStatus.textContent = 'Waiting for a face to appear...';
                            recognizedName.textContent = '';
                            confidenceContainer.style.display = 'none';
                            
                            // Restart monitoring after a proper delay
                            setTimeout(() => {
                                detectionCooldown = false;
                                // Will automatically restart face monitoring
                            }, 1000);
                        }
                    }
                }, 3000);
                
            } else {
                // Handle non-match cases
                if (data.id === "No Face") {
                    matchText.textContent = 'NO FACE DETECTED';
                    matchText.className = 'match-text no-match';
                    recognitionStatus.textContent = 'No face detected. Waiting for a face to appear...';
                    
                    // Go back to face monitoring mode
                    isScanningPaused = true;
                    
                    if (faceDetectionTimer) {
                        clearInterval(faceDetectionTimer);
                        faceDetectionTimer = null;
                    }
                    
                    // Resume monitoring after short delay with clean buffer
                    setTimeout(() => {
                        matchText.textContent = 'WAITING FOR FACE';
                        matchText.className = 'match-text';
                        recognizing = false;
                        detectionCooldown = false;
                        faceDetectionBuffer = [];
                    }, 1500);
                    
                } else if (data.id === "Low Confidence") {
                    matchText.textContent = 'LOW CONFIDENCE';
                    matchText.className = 'match-text no-match';
                    recognitionStatus.textContent = 'Face detected but confidence too low.';
                    
                    // Display result for 2 seconds before resuming scanning
                    resultDisplayTimer = setTimeout(() => {
                        // Release the lock
                        recognizing = false;
                        
                        // Check if face is still present
                        const checkContext = canvas.getContext('2d');
                        canvas.width = 320;
                        canvas.height = 240;
                        checkContext.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const checkImageData = checkContext.getImageData(0, 0, canvas.width, canvas.height);
                        
                        if (hasFace(checkImageData)) {
                            console.log("Face still present after low confidence - continuing scan");
                            // Reset buffer but keep scanning
                            faceDetectionBuffer = [];
                            isScanningPaused = false;
                            
                            matchText.textContent = 'SCANNING';
                            matchText.className = 'match-text';
                            recognitionStatus.textContent = 'Continuing to scan...';
                            
                            // Restart active detection
                            if (!faceDetectionTimer) {
                                startFaceDetection();
                            }
                            
                            // Increase cooldown time to prevent too frequent recognitions
                            setTimeout(() => {
                                detectionCooldown = false;
                            }, 2000); // Increased from 500ms to 2000ms
                        } else {
                            // Reset detection buffer for a fresh start
                            faceDetectionBuffer = [];
                            
                            // Reset scanning state
                            isScanningPaused = true;
                            
                            if (faceDetectionTimer) {
                                clearInterval(faceDetectionTimer);
                                faceDetectionTimer = null;
                            }
                            
                            matchText.textContent = 'WAITING FOR FACE';
                            matchText.className = 'match-text';
                            recognitionStatus.textContent = 'Waiting for a face to appear...';
                            
                            // Resume detection with a delay
                            setTimeout(() => {
                                detectionCooldown = false;
                            }, 800);
                        }
                    }, 2000);
                    
                } else {
                    // Unknown person
                    matchText.textContent = 'ACCESS DENIED';
                    matchText.className = 'match-text no-match';
                    recognitionStatus.textContent = 'Unknown person detected.';
                    
                    // Record the access denial if not already recorded
                    if (!data.recorded) {
                        recordAccessFailure();
                    }
                    
                    // Display result for 3 seconds before resuming scanning
                    resultDisplayTimer = setTimeout(() => {
                        // Release lock
                        recognizing = false;
                        
                        // Check if face is still present
                        const checkContext = canvas.getContext('2d');
                        canvas.width = 320;
                        canvas.height = 240;
                        checkContext.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const checkImageData = checkContext.getImageData(0, 0, canvas.width, canvas.height);
                        
                        if (hasFace(checkImageData)) {
                            console.log("Face still present after unknown person - continuing scan");
                            // Reset buffer but keep scanning
                            faceDetectionBuffer = [];
                            isScanningPaused = false;
                            
                            matchText.textContent = 'SCANNING';
                            matchText.className = 'match-text';
                            recognitionStatus.textContent = 'Continuing to scan...';
                            
                            // Restart active detection
                            if (!faceDetectionTimer) {
                                startFaceDetection();
                            }
                            
                            // Increase cooldown time to prevent too frequent recognitions
                            setTimeout(() => {
                                detectionCooldown = false;
                            }, 2000); // Increased from 500ms to 2000ms
                        } else {
                            // Reset detection buffer for a fresh start
                            faceDetectionBuffer = [];
                            
                            // Reset scanning state
                            isScanningPaused = true;
                            
                            if (faceDetectionTimer) {
                                clearInterval(faceDetectionTimer);
                                faceDetectionTimer = null;
                            }
                            
                            matchText.textContent = 'WAITING FOR FACE';
                            matchText.className = 'match-text';
                            recognitionStatus.textContent = 'Waiting for a face to appear...';
                            
                            // Resume detection after a longer delay
                            setTimeout(() => {
                                detectionCooldown = false;
                            }, 1000);
                        }
                    }, 3000);
                }
            }
        })
        .catch(error => {
            processingIndicator.style.display = 'none';
            
            // Stop progress animation with failure
            stopProgressAnimation(false);
            
            console.error('Recognition error:', error);
            errorMessage.textContent = error.message || 'Error processing face. Please try again.';
            errorMessage.style.display = 'block';
            matchText.textContent = 'ERROR';
            matchText.className = 'match-text no-match';
            
            // Hide error message after a few seconds
            setTimeout(() => {
                errorMessage.style.display = 'none';
                
                // Reset detection buffer for a fresh start
                faceDetectionBuffer = [];
                
                // Go back to face monitoring mode
                isScanningPaused = true;
                
                if (faceDetectionTimer) {
                    clearInterval(faceDetectionTimer);
                    faceDetectionTimer = null;
                }
                
                matchText.textContent = 'WAITING FOR FACE';
                matchText.className = 'match-text';
                recognitionStatus.textContent = 'Waiting for a face to appear...';
                
                // Resume detection after error recovery
                recognizing = false;
                setTimeout(() => {
                    detectionCooldown = false;
                }, 1500);
            }, 3000);
        });
    }
    catch (error) {
        // Hide processing indicator
        processingIndicator.style.display = 'none';
        
        // Stop progress animation with failure
        stopProgressAnimation(false);
        
        console.error('Capture error:', error);
        errorMessage.textContent = 'Error capturing image. Please check your camera.';
        errorMessage.style.display = 'block';
        
        // Hide error message after a few seconds
        setTimeout(() => {
            errorMessage.style.display = 'none';
            
            // Reset detection buffer for a fresh start
            faceDetectionBuffer = [];
            
            // Go back to face monitoring mode
            isScanningPaused = true;
            
            if (faceDetectionTimer) {
                clearInterval(faceDetectionTimer);
                faceDetectionTimer = null;
            }
            
            matchText.textContent = 'WAITING FOR FACE';
            matchText.className = 'match-text';
            
            // Resume detection after recovery
            recognizing = false;
            setTimeout(() => {
                detectionCooldown = false;
            }, 1500);
        }, 3000);
    }
}

// When video starts playing, do an immediate face check
video.addEventListener('play', function() {
    // Wait a short moment for camera to stabilize
    setTimeout(() => {
        if (isActive && isScanningPaused) {
            // Run multiple checks with short intervals
            for (let i = 0; i < 5; i++) {
                setTimeout(checkForFacePresence, i * 200);
            }
        }
    }, 1000);
});
