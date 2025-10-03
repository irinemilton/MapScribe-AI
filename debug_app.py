#!/usr/bin/env python3
"""
Debug version of MapScribe AI with enhanced error handling
"""

import os
import json
import uuid
import traceback
from datetime import datetime
from pathlib import Path
import logging

from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import image processing libraries with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV imported successfully")
except ImportError as e:
    CV2_AVAILABLE = False
    logger.warning(f"⚠️ OpenCV not available: {e}")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("✅ NumPy imported successfully")
except ImportError as e:
    NUMPY_AVAILABLE = False
    logger.warning(f"⚠️ NumPy not available: {e}")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("✅ PIL imported successfully")
except ImportError as e:
    PIL_AVAILABLE = False
    logger.warning(f"⚠️ PIL not available: {e}")

# Simple fallback processor that always works
class DebugProcessor:
    """Debug processor that provides detailed error information."""
    
    def __init__(self):
        self.name = "Debug Processor"
        self.version = "1.0.0"
        logger.info(f"🔧 Initialized {self.name}")
    
    def process_image(self, image_path, options=None):
        """Process image with detailed logging."""
        start_time = datetime.now()
        logger.info(f"🔄 Processing image: {image_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                error_msg = f"File not found: {image_path}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'processing_time': 0
                }
            
            # Get file info
            file_size = os.path.getsize(image_path)
            logger.info(f"📊 File size: {file_size} bytes")
            
            # Try to get image dimensions
            width, height = None, None
            
            if CV2_AVAILABLE:
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        height, width = image.shape[:2]
                        logger.info(f"📐 Image dimensions (OpenCV): {width}x{height}")
                except Exception as e:
                    logger.warning(f"OpenCV failed: {e}")
            
            if (width is None or height is None) and PIL_AVAILABLE:
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        logger.info(f"📐 Image dimensions (PIL): {width}x{height}")
                except Exception as e:
                    logger.warning(f"PIL failed: {e}")
            
            # Generate demo results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'success': True,
                'total_detections': 8,
                'categorized_results': {
                    'cities': [
                        {'display_name': 'Mumbai', 'confidence': 0.95, 'original_text': 'Mumbai'},
                        {'display_name': 'Delhi', 'confidence': 0.92, 'original_text': 'Delhi'}
                    ],
                    'states': [
                        {'display_name': 'Maharashtra', 'confidence': 0.88, 'original_text': 'Maharashtra'},
                        {'display_name': 'Karnataka', 'confidence': 0.85, 'original_text': 'Karnataka'}
                    ],
                    'rivers': [
                        {'display_name': 'Ganges', 'confidence': 0.90, 'original_text': 'Ganga'},
                        {'display_name': 'Yamuna', 'confidence': 0.87, 'original_text': 'Yamuna'}
                    ]
                },
                'summary': {
                    'total_places': 6,
                    'validated_places': 6,
                    'categories': {
                        'cities': {'count': 2},
                        'states': {'count': 2},
                        'rivers': {'count': 2}
                    }
                },
                'processing_time': processing_time,
                'processor_info': {
                    'name': self.name,
                    'version': self.version,
                    'cv2_available': CV2_AVAILABLE,
                    'numpy_available': NUMPY_AVAILABLE,
                    'pil_available': PIL_AVAILABLE
                },
                'image_info': {
                    'width': width or 1000,
                    'height': height or 800,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                },
                'demo_note': 'Debug processor - simulated results for testing'
            }
            
            logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

# Initialize processor
processor = DebugProcessor()

# Flask app configuration
app = Flask(__name__)
CORS(app)
app.secret_key = 'mapscribe-debug-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static/demo', exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(filepath):
    """Get basic information about an uploaded file."""
    try:
        file_size = os.path.getsize(filepath)
        
        # Try different methods to get image dimensions
        width, height = None, None
        
        if CV2_AVAILABLE:
            try:
                image = cv2.imread(filepath)
                if image is not None:
                    height, width = image.shape[:2]
            except:
                pass
        
        if (width is None or height is None) and PIL_AVAILABLE:
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except:
                pass
        
        return {
            'width': width or 1000,
            'height': height or 800,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {
            'width': 1000,
            'height': 800,
            'size_bytes': 0,
            'size_mb': 0
        }

@app.route('/')
def index():
    """Main landing page."""
    logger.info("📄 Serving index page")
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing with detailed error logging."""
    if request.method == 'GET':
        return render_template('upload.html')
    
    logger.info("🔄 Processing upload request")
    
    try:
        # Log request details
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form: {dict(request.form)}")
        
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
            return jsonify({'error': 'File type not supported. Please upload PNG, JPG, TIFF, or BMP files.'}), 400
        
        # Get processing options
        task = request.form.get('task', 'task4')
        confidence_threshold = float(request.form.get('confidence', 0.5))
        language = request.form.get('language', 'auto')
        
        logger.info(f"Processing options: task={task}, confidence={confidence_threshold}, language={language}")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            logger.error("File was not saved successfully")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        logger.info(f"File saved successfully: {os.path.getsize(filepath)} bytes")
        
        # Get file information
        file_info = get_file_info(filepath)
        logger.info(f"File info: {file_info}")
        
        # Process the image
        processing_options = {
            'task': task,
            'confidence_threshold': max(0.1, confidence_threshold),
            'language': language
        }
        
        logger.info("🔄 Starting image processing")
        results = processor.process_image(filepath, processing_options)
        logger.info(f"Processing results: success={results.get('success', False)}")
        
        if not results.get('success', False):
            logger.error(f"Processing failed: {results.get('error', 'Unknown error')}")
            return jsonify({
                'error': f'Processing failed: {results.get("error", "Unknown error")}',
                'details': results
            }), 500
        
        # Save results
        results_filename = f"{file_id}_results.json"
        results_path = os.path.join(RESULTS_FOLDER, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Prepare response
        response_data = {
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'file_info': file_info,
            'processing_options': processing_options,
            'results': results,
            'image_url': f'/static/uploads/{unique_filename}',
            'results_url': f'/static/results/{results_filename}',
            'detections_count': results.get('total_detections', 0),
            'recognized_count': results.get('summary', {}).get('total_places', 0),
            'processing_time': results.get('processing_time', 0),
            'session_id': file_id,
            'smart_results': True,
            'smart_results_url': f'/smart-results/{file_id}'
        }
        
        logger.info("✅ Upload processing completed successfully")
        return jsonify(response_data)
        
    except RequestEntityTooLarge:
        logger.error("File too large")
        return jsonify({'error': 'File too large. Maximum size is 64MB.'}), 413
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'debug_info': {
                'cv2_available': CV2_AVAILABLE,
                'numpy_available': NUMPY_AVAILABLE,
                'pil_available': PIL_AVAILABLE,
                'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
                'results_folder_exists': os.path.exists(RESULTS_FOLDER)
            }
        }), 500

@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return jsonify({
        'status': 'online',
        'version': '1.0.0-debug',
        'name': 'MapScribe AI Debug',
        'processor_type': 'debug',
        'environment': os.environ.get('FLASK_ENV', 'development'),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 64,
        'dependencies': {
            'cv2_available': CV2_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'pil_available': PIL_AVAILABLE
        },
        'directories': {
            'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
            'results_folder_exists': os.path.exists(RESULTS_FOLDER)
        }
    })

@app.route('/health')
def health_check():
    """Simple health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'processor': processor.name
    })

@app.route('/debug')
def debug_info():
    """Debug information page."""
    return jsonify({
        'system_info': {
            'python_version': os.sys.version,
            'working_directory': os.getcwd(),
            'environment_variables': dict(os.environ)
        },
        'dependencies': {
            'cv2_available': CV2_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'pil_available': PIL_AVAILABLE
        },
        'directories': {
            'upload_folder': UPLOAD_FOLDER,
            'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
            'results_folder': RESULTS_FOLDER,
            'results_folder_exists': os.path.exists(RESULTS_FOLDER)
        },
        'processor_info': {
            'name': processor.name,
            'version': processor.version
        }
    })

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.url}")
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return render_template('error.html',
                         error_code=500, 
                         error_message="Internal server error"), 500

@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors."""
    logger.warning("File too large error")
    return render_template('error.html',
                         error_code=413,
                         error_message="File too large. Maximum size is 64MB."), 413

if __name__ == '__main__':
    print("🐛 Starting MapScribe AI Debug Server...")
    print("=" * 50)
    print("🌐 Web Interface: http://localhost:5000")
    print("📊 API Status: http://localhost:5000/api/status")
    print("🐛 Debug Info: http://localhost:5000/debug")
    print("💚 Health Check: http://localhost:5000/health")
    print("=" * 50)
    
    # Run the Flask app with debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)