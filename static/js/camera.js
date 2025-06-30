// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('startButton');
const captureButton = document.getElementById('captureButton');
const saveButton = document.getElementById('saveButton');
const matchOverlay = document.getElementById('match-overlay');
const matchText = document.getElementById('match-text');
const recognitionStatus = document.getElementById('recognitionStatus');
const recognizedName = document.getElementById('recognizedName');
const confidenceLevel = document.getElementById('confidenceLevel');
const saveReferenceForm = document.getElementById('saveReferenceForm');
const referenceName = document.getElementById('referenceName');
const confirmSaveButton = document.getElementById('confirmSaveButton');
const cancelSaveButton = document.getElementById('cancelSaveButton');

// Global variables
let stream = null;
let capturedImage = null;

// Event listeners
startButton.addEventListener('click', startCamera);
captureButton.addEventListener('click', captureAndRecognize);
saveButton.addEventListener('click', showSaveReferenceForm);
confirmSaveButton.addEventListener('click', saveReference);
cancelSaveButton.addEventListener('click', hideSaveReferenceForm);

// Start the camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480,
                facingMode: 'user'
            } 
        });
        
        video.srcObject = stream;
        
        // Enable capture button once camera is started
        startButton.disabled = true;
        captureButton.disabled = false;
        
        // Reset other display elements
        recognizedName.textContent = '';
        confidenceLevel.textContent = '';
        matchText.textContent = 'READY';
        matchText.className = 'match-text';
        
    } catch (err) {
        console.error('Error accessing camera:', err);
    }
}

// Capture image and send for recognition
function captureAndRecognize() {
    if (!stream) {
        return;
    }
    
    // Get the canvas context
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the current video frame on the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64 image
    capturedImage = canvas.toDataURL('image/jpeg');
    
    // Update UI to show processing
    matchText.textContent = 'PROCESSING...';
    matchText.className = 'match-text';
    
    // Send to server for recognition
    fetch('/process_face', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: capturedImage
        })
    })
    .then(response => response.json())
    .then(data => {
        // Enable save button
        saveButton.disabled = false;
        
        // Update UI with recognition results
        if (data.match) {
            recognizedName.textContent = `Name: ${data.name}`;
            confidenceLevel.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
            
            matchText.textContent = 'MATCH';
            matchText.className = 'match-text match';
        } else {
            recognizedName.textContent = '';
            confidenceLevel.textContent = '';
            
            matchText.textContent = 'NO MATCH';
            matchText.className = 'match-text no-match';
        }
    })
    .catch(error => {
        console.error('Error processing face:', error);
    });
}

// Show form to save reference image
function showSaveReferenceForm() {
    if (!capturedImage) {
        return;
    }
    
    saveReferenceForm.style.display = 'block';
    referenceName.focus();
}

// Hide the save reference form
function hideSaveReferenceForm() {
    saveReferenceForm.style.display = 'none';
    referenceName.value = '';
}

// Save reference image
function saveReference() {
    const name = referenceName.value.trim();
    
    if (!name) {
        alert('Please enter a name for the reference image.');
        return;
    }
    
    if (!capturedImage) {
        alert('No image captured. Please capture an image first.');
        return;
    }
    
    // Send reference image to server
    fetch('/save_reference', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: capturedImage,
            name: name
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            hideSaveReferenceForm();
        }
    })
    .catch(error => {
        console.error('Error saving reference:', error);
    });
}

// Clean up resources when leaving the page
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
