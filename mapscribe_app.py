#!/usr/bin/env python3
"""
MapScribe AI - Flask Web Application
A beautiful, dynamic web interface for historical map text processing.
Upload historical maps and discover place names with AI-powered analysis.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging

from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import cv2
import numpy as np

# Import our image processing system with robust fallback
PROCESSOR_TYPE = "unknown"
MapScribeProcessor = None

# Try different processors in order of preference
processors_to_try = [
    # Production/Cloud processors
    ("render_processor", "RenderProcessor", "render_optimized"),
    # Development processors
    ("json_safe_processor", "JSONSafeProcessor", "json_safe"),
    ("working_processor", "WorkingProcessor", "working_smart"),
    # Fallback processor
    ("simple_processor", "SimpleMapProcessor", "simple")
]

for module_name, class_name, processor_type in processors_to_try:
    try:
        module = __import__(module_name)
        MapScribeProcessor = getattr(module, class_name)
        PROCESSOR_TYPE = processor_type
        break
    except (ImportError, AttributeError) as e:
        continue

# Final fallback - create a minimal processor if nothing works
if MapScribeProcessor is None:
    class MinimalProcessor:
        def __init__(self):
            self.name = "Minimal Fallback Processor"
        
        def process_image(self, image_path, options=None):
            return {
                'success': True,
                'total_detections': 0,
                'categorized_results': {},
                'summary': {'total_places': 0},
                'processing_time': 0.1,
                'error': 'Using minimal fallback processor - no real processing available'
            }
    
    MapScribeProcessor = MinimalProcessor
    PROCESSOR_TYPE = "minimal_fallback"

# Flask app configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'mapscribe-ai-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static/demo', exist_ok=True)

# Initialize the processor
processor = MapScribeProcessor()
print(f"🔧 Using {PROCESSOR_TYPE} processor")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info(filepath):
    """Get basic information about an uploaded file."""
    try:
        image = cv2.imread(filepath)
        if image is None:
            return None
        
        height, width = image.shape[:2]
        file_size = os.path.getsize(filepath)
        
        return {
            'width': width,
            'height': height,
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return None


@app.route('/')
def index():
    """Main landing page with hero section and upload interface."""
    return render_template('index.html')


@app.route('/demo')
def demo():
    """Demo page with sample historical maps."""
    demo_maps = [
        {
            'name': 'London Historical Map',
            'filename': 'demo_london.jpg',
            'description': 'Victorian-era map of London showing major landmarks and districts',
            'year': '1890s',
            'region': 'London, England'
        },
        {
            'name': 'Paris Ancient Map',
            'filename': 'demo_paris.jpg', 
            'description': 'Medieval map of Paris with historical place names',
            'year': '1650s',
            'region': 'Paris, France'
        },
        {
            'name': 'New York Harbor Map',
            'filename': 'demo_newyork.jpg',
            'description': 'Early American map showing Manhattan and surrounding areas',
            'year': '1770s',
            'region': 'New York, USA'
        }
    ]
    return render_template('demo.html', demo_maps=demo_maps)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing."""
    if request.method == 'GET':
        return render_template('upload.html')
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported. Please upload PNG, JPG, TIFF, or BMP files.'}), 400
        
        # Get processing options
        task = request.form.get('task', 'task4')
        confidence_threshold = float(request.form.get('confidence', 0.5))
        language = request.form.get('language', 'auto')
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Get file information
        file_info = get_file_info(filepath)
        if not file_info:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process the image with timeout handling
        logger.info(f"Processing image: {filename} with task: {task}")
        
        processing_options = {
            'task': task,
            'confidence_threshold': max(0.1, confidence_threshold),  # Lower threshold for better results
            'language': language
        }
        
        # Process with optimized fast processor
        try:
            results = processor.process_image(filepath, processing_options)
            if not results.get('success', False):
                raise Exception(results.get('error', 'Processing failed'))
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        # Save results
        results_filename = f"{file_id}_results.json"
        results_path = os.path.join(RESULTS_FOLDER, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Prepare response with expected fields for frontend
        response_data = {
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'file_info': file_info,
            'processing_options': processing_options,
            'results': results,
            'image_url': f'/static/uploads/{unique_filename}',
            'results_url': f'/static/results/{results_filename}',
            # Add fields expected by frontend
            'detections_count': results.get('total_detections', 0),
            'recognized_count': results.get('summary', {}).get('total_places', 0),
            'processing_time': results.get('processing_time', 0),
            'session_id': file_id,
            'smart_results': True,  # Flag for smart categorized results
            'smart_results_url': f'/smart-results/{file_id}'  # URL for detailed view
        }
        
        return jsonify(response_data)
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
    
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/process-demo/<demo_name>')
def process_demo(demo_name):
    """Process a demo image."""
    try:
        # Demo image mapping
        demo_files = {
            'london': 'demo_historical_map.jpg',  # Use our existing demo
            'paris': 'demo_historical_map.jpg',   # Placeholder
            'newyork': 'demo_historical_map.jpg'  # Placeholder
        }
        
        if demo_name not in demo_files:
            return jsonify({'error': 'Demo not found'}), 404
        
        demo_filename = demo_files[demo_name]
        demo_path = demo_filename  # Assuming it's in the root directory
        
        if not os.path.exists(demo_path):
            return jsonify({'error': 'Demo file not found'}), 404
        
        # Process with default options optimized for demo
        processing_options = {
            'task': 'task4',
            'confidence_threshold': 0.3,  # Lower threshold for demo
            'language': 'auto'
        }
        
        results = processor.process_image(demo_path, processing_options)
        
        # Get file info
        file_info = get_file_info(demo_path)
        
        response_data = {
            'success': True,
            'demo_name': demo_name,
            'filename': demo_filename,
            'file_info': file_info,
            'processing_options': processing_options,
            'results': results,
            'image_url': f'/{demo_filename}',
            'is_demo': True
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing demo: {e}")
        return jsonify({'error': f'Demo processing failed: {str(e)}'}), 500


@app.route('/results/<file_id>')
def view_results(file_id):
    """View detailed results for a processed image."""
    try:
        results_path = os.path.join(RESULTS_FOLDER, f"{file_id}_results.json")
        
        if not os.path.exists(results_path):
            flash('Results not found', 'error')
            return redirect(url_for('index'))
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Find the original image
        image_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                image_path = f"/static/uploads/{file_id}.{ext}"
                break
        
        return render_template('results.html', 
                             results=results, 
                             image_url=image_path,
                             file_id=file_id)
        
    except Exception as e:
        logger.error(f"Error viewing results: {e}")
        flash('Error loading results', 'error')
        return redirect(url_for('index'))


@app.route('/smart-results/<file_id>')
def view_smart_results(file_id):
    """View smart categorized results for a processed image."""
    try:
        results_path = os.path.join(RESULTS_FOLDER, f"{file_id}_results.json")
        
        if not os.path.exists(results_path):
            flash('Results not found', 'error')
            return redirect(url_for('index'))
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Find the original image
        image_path = None
        for ext in ALLOWED_EXTENSIONS:
            potential_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{ext}")
            if os.path.exists(potential_path):
                image_path = f"/static/uploads/{file_id}.{ext}"
                break
        
        return render_template('smart_results.html', 
                             results=results, 
                             image_url=image_path,
                             file_id=file_id)
        
    except Exception as e:
        logger.error(f"Error viewing smart results: {e}")
        flash('Error loading results', 'error')
        return redirect(url_for('index'))


@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'name': 'MapScribe AI',
        'processor_type': PROCESSOR_TYPE,
        'environment': os.environ.get('FLASK_ENV', 'development'),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': 64,
        'available_tasks': {
            'task1': 'Word Detection',
            'task2': 'Phrase Detection',
            'task3': 'Word Detection + Recognition',
            'task4': 'Phrase Detection + Recognition (Recommended)'
        }
    })

@app.route('/health')
def health_check():
    """Simple health check for monitoring services."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/about')
def about():
    """About page with information about MapScribe AI."""
    return render_template('about.html')


@app.route('/help')
def help_page():
    """Help page with usage instructions."""
    return render_template('help.html')


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html',
                         error_code=500, 
                         error_message="Internal server error"), 500


@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors."""
    return render_template('error.html',
                         error_code=413,
                         error_message="File too large. Maximum size is 16MB."), 413


if __name__ == '__main__':
    print("🗺️  Starting MapScribe AI...")
    print("=" * 50)
    print("🌐 Web Interface: http://localhost:5000")
    print("📊 API Status: http://localhost:5000/api/status")
    print("🎯 Demo Maps: http://localhost:5000/demo")
    print("=" * 50)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)