# Secure Facial Recognition Access Control For Private Office

This project is a secure facial recognition-based access control system designed for private office environments. It leverages deep learning and computer vision to authenticate users and manage access records.

## Features
- Employee management (add, edit, delete)
- Real-time facial recognition
- Access logging and reporting
- Admin authentication and password management
- Web-based interface (Flask)

## Requirements
- **Python 3.10+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Supported OS:** Windows 10/11, Linux, macOS

## Python Libraries
All required libraries and their versions are listed in `requirements.txt`. Main dependencies:
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- deepface >= 0.0.79
- ultralytics >= 8.0.196
- scikit-image >= 0.21.0
- onnxruntime >= 1.15.0
- python-dotenv >= 0.19.0
- mediapipe >= 0.10.0
- insightface >= 0.7.3
- scipy == 1.10.1
- flask >= 2.0.1
- imutils == 0.5.4

Install all dependencies with:
```
pip install -r requirements.txt
```

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone https://github.com/cocobliss55/Secure-Facial-Recognition-Access-Control-For-Private-Office.git
   cd Secure-Facial-Recognition-Access-Control-For-Private-Office
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Prepare models and data:**
   - Ensure the `models/` and `static/references/` folders contain the required model files and reference images.
   - The `.env` file should be present for environment variables (if used).

4. **Run the application:**
   ```
   python app.py
   ```
   The web interface will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Usage
- **Login:** Use the admin credentials to log in.
   ID: admin
   Password: 1211103616
- **Add Employees:** Add new employees and their reference images via the web interface.
- **Recognition:** Use the recognition page to perform real-time facial recognition.
- **Reports:** View access logs and reports from the dashboard.

## Notes
- For best results, use high-quality reference images for each employee.
- The system uses YOLO and other deep learning models for face detection and recognition.
- All access attempts are logged for security and auditing.

## License
This project is for educational and private use. For commercial use, please contact the author.

---

For more details, see the source code and comments.

GitHub Repository: https://github.com/cocobliss55/Secure-Facial-Recognition-Access-Control-For-Private-Office 