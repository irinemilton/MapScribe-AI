#!/usr/bin/env python3
"""
Minimal MapScribe AI - No dependencies version
"""

import os
import json
import uuid
import traceback
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'minimal-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({
        'message': 'MapScribe AI Minimal Version',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Basic validation
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Save file
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Create mock results
        results = {
            'success': True,
            'total_detections': 3,
            'categorized_results': {
                'cities': [
                    {'display_name': 'Mumbai', 'confidence': 0.95}
                ],
                'states': [
                    {'display_name': 'Maharashtra', 'confidence': 0.90}
                ],
                'rivers': [
                    {'display_name': 'Ganges', 'confidence': 0.85}
                ]
            },
            'summary': {
                'total_places': 3,
                'validated_places': 3
            },
            'processing_time': 0.5,
            'file_info': {
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2)
            },
            'note': 'Minimal processor - demo results'
        }
        
        # Save results
        results_filename = f"{file_id}_results.json"
        results_path = os.path.join(RESULTS_FOLDER, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'results': results,
            'image_url': f'/static/uploads/{unique_filename}',
            'results_url': f'/static/results/{results_filename}'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Upload failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'version': '1.0.0-minimal',
        'name': 'MapScribe AI Minimal',
        'supported_formats': list(ALLOWED_EXTENSIONS)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.errorhandler(500)
def handle_500(e):
    return jsonify({
        'error': 'Internal server error',
        'details': str(e),
        'traceback': traceback.format_exc()
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Minimal MapScribe AI...")
    app.run(debug=True, host='0.0.0.0', port=5000)