#!/usr/bin/env python3
"""
Install Flask-CORS for web interface
"""

import subprocess
import sys

def install_flask_cors():
    """Install Flask-CORS package."""
    try:
        print("📦 Installing Flask-CORS...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-cors"])
        print("✅ Flask-CORS installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Flask-CORS: {e}")
        return False

if __name__ == "__main__":
    success = install_flask_cors()
    if success:
        print("\n🚀 Ready to run the web interface!")
        print("Run: python mapscribe_app.py")
    else:
        print("\n⚠️ Installation failed. Try manually:")
        print("pip install flask-cors")