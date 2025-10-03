#!/usr/bin/env python3
"""
Start MapScribe AI Web Server
Optimized startup script with better error handling
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import flask
        import cv2
        import numpy as np
        from flask_cors import CORS
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_directories():
    """Create required directories."""
    dirs = ['static/uploads', 'static/results', 'static/demo']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directories created")

def start_server():
    """Start the Flask server."""
    try:
        print("🗺️  Starting MapScribe AI Web Server...")
        print("=" * 50)
        print("🌐 Web Interface: http://localhost:5000")
        print("📊 API Status: http://localhost:5000/api/status")
        print("🎯 Demo Maps: http://localhost:5000/demo")
        print("=" * 50)
        print("💡 Tips:")
        print("   - Use smaller images (< 5MB) for faster processing")
        print("   - Try the demo maps if you don't have historical maps")
        print("   - Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Import and run the Flask app
        from mapscribe_app import app
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure port 5000 is not in use")
        print("2. Check if all dependencies are installed")
        print("3. Try running: python mapscribe_app.py directly")

def main():
    """Main startup function."""
    print("🚀 MapScribe AI Startup")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies:")
        print("pip install flask flask-cors opencv-python numpy")
        return
    
    # Create directories
    create_directories()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()