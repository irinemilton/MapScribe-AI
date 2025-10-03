#!/usr/bin/env python3
"""
Ultra minimal Flask app - no external dependencies
"""

import os
import json
import uuid
from datetime import datetime

# Try to import Flask - this might be the issue
try:
    from flask import Flask, request, jsonify
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")
    exit(1)

# Try werkzeug
try:
    from werkzeug.utils import secure_filename
    print("✅ Werkzeug imported successfully")
except ImportError as e:
    print(f"❌ Werkzeug import failed: {e}")
    # Create a simple fallback
    def secure_filename(filename):
        return filename.replace('/', '_').replace('\\', '_')

# Skip CORS for now
app = Flask(__name__)
app.secret_key = 'test-key'

# Simple directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({
        'message': 'Ultra Minimal MapScribe AI',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test')
def test():
    return jsonify({'test': 'working'})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Log everything
        print(f"Request method: {request.method}")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request form: {dict(request.form)}")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file in request'}), 400
        
        file = request.files['file']
        print(f"File: {file.filename}")
        
        if not file.filename:
            return jsonify({'error': 'No filename'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        
        print(f"Saving to: {filepath}")
        file.save(filepath)
        
        # Check if saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"File saved: {file_size} bytes")
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'size': file_size,
                'path': filepath
            })
        else:
            return jsonify({'error': 'File not saved'}), 500
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.errorhandler(500)
def handle_error(e):
    print(f"500 Error: {e}")
    return jsonify({'error': 'Server error', 'details': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Ultra Minimal App...")
    print("Test URLs:")
    print("- http://localhost:5000/")
    print("- http://localhost:5000/test")
    print("- POST http://localhost:5000/upload")
    
    app.run(debug=True, host='0.0.0.0', port=5000)