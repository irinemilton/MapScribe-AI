#!/usr/bin/env python3
"""
Logging version to catch the exact error
"""

import os
import sys
import logging
import traceback
from datetime import datetime

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

logger.info("🚀 Starting application...")

# Test all imports one by one
try:
    import json
    logger.info("✅ json imported")
except Exception as e:
    logger.error(f"❌ json import failed: {e}")

try:
    import uuid
    logger.info("✅ uuid imported")
except Exception as e:
    logger.error(f"❌ uuid import failed: {e}")

try:
    from flask import Flask, render_template, request, jsonify
    logger.info("✅ Flask imported")
except Exception as e:
    logger.error(f"❌ Flask import failed: {e}")
    sys.exit(1)

try:
    from flask_cors import CORS
    logger.info("✅ Flask-CORS imported")
    CORS_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Flask-CORS import failed: {e}")
    CORS_AVAILABLE = False

try:
    from werkzeug.utils import secure_filename
    logger.info("✅ Werkzeug imported")
except Exception as e:
    logger.error(f"❌ Werkzeug import failed: {e}")
    def secure_filename(filename):
        return filename

try:
    from werkzeug.exceptions import RequestEntityTooLarge
    logger.info("✅ Werkzeug exceptions imported")
except Exception as e:
    logger.warning(f"⚠️ Werkzeug exceptions import failed: {e}")
    class RequestEntityTooLarge(Exception):
        pass

# Try optional imports
try:
    import cv2
    logger.info("✅ OpenCV imported")
    CV2_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ OpenCV not available: {e}")
    CV2_AVAILABLE = False

try:
    import numpy as np
    logger.info("✅ NumPy imported")
    NUMPY_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ NumPy not available: {e}")
    NUMPY_AVAILABLE = False

# Create Flask app
logger.info("Creating Flask app...")
app = Flask(__name__)

if CORS_AVAILABLE:
    CORS(app)
    logger.info("✅ CORS enabled")

app.secret_key = 'logging-test-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

# Create directories
logger.info("Creating directories...")
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    logger.info(f"✅ Directories created: {UPLOAD_FOLDER}, {RESULTS_FOLDER}")
except Exception as e:
    logger.error(f"❌ Directory creation failed: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("📄 Index route accessed")
    return jsonify({
        'message': 'MapScribe AI Logging Version',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'imports': {
            'cors_available': CORS_AVAILABLE,
            'cv2_available': CV2_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        }
    })

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    logger.info(f"📤 Upload route accessed - Method: {request.method}")
    
    if request.method == 'GET':
        return jsonify({'message': 'Upload endpoint ready', 'method': 'GET'})
    
    try:
        logger.info("🔍 Checking request...")
        logger.info(f"Request files keys: {list(request.files.keys())}")
        logger.info(f"Request form keys: {list(request.form.keys())}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not supported'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        logger.info(f"💾 Saving file to: {filepath}")
        
        file.save(filepath)
        
        # Verify file was saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            logger.info(f"✅ File saved successfully: {file_size} bytes")
        else:
            logger.error("❌ File was not saved")
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Create simple results
        results = {
            'success': True,
            'total_detections': 5,
            'categorized_results': {
                'cities': [{'display_name': 'Test City', 'confidence': 0.9}],
                'states': [{'display_name': 'Test State', 'confidence': 0.8}]
            },
            'summary': {'total_places': 2, 'validated_places': 2},
            'processing_time': 0.1,
            'file_info': {
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }
        }
        
        # Save results
        results_filename = f"{file_id}_results.json"
        results_path = os.path.join(RESULTS_FOLDER, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to: {results_path}")
        
        response_data = {
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'results': results,
            'image_url': f'/static/uploads/{unique_filename}',
            'results_url': f'/static/results/{results_filename}'
        }
        
        logger.info("✅ Upload completed successfully")
        return jsonify(response_data)
        
    except RequestEntityTooLarge:
        logger.error("File too large")
        return jsonify({'error': 'File too large'}), 413
    
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/status')
def api_status():
    logger.info("📊 Status endpoint accessed")
    return jsonify({
        'status': 'online',
        'version': '1.0.0-logging',
        'name': 'MapScribe AI Logging',
        'imports': {
            'cors_available': CORS_AVAILABLE,
            'cv2_available': CV2_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        },
        'directories': {
            'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
            'results_folder_exists': os.path.exists(RESULTS_FOLDER)
        }
    })

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found', 'url': request.url}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        'error': 'Internal server error',
        'details': str(error),
        'traceback': traceback.format_exc()
    }), 500

if __name__ == '__main__':
    logger.info("🌐 Starting Flask server...")
    print("🚀 MapScribe AI Logging Version")
    print("📋 Check app.log for detailed logs")
    print("🌐 Server: http://localhost:5000")
    print("📊 Status: http://localhost:5000/api/status")
    
    app.run(debug=True, host='0.0.0.0', port=5000)