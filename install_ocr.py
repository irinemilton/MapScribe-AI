#!/usr/bin/env python3
"""
Installation script for OCR dependencies.
This will install the necessary packages for real text recognition.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def install_package(package_name):
    """Install a Python package using pip."""
    print(f"📦 Installing {package_name}...")
    success, output = run_command(f"{sys.executable} -m pip install {package_name}")
    
    if success:
        print(f"✅ {package_name} installed successfully!")
        return True
    else:
        print(f"❌ Failed to install {package_name}: {output}")
        return False

def check_package(package_name):
    """Check if a package is already installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Main installation function."""
    print("🗺️  MapScribe AI - OCR Dependencies Installation")
    print("=" * 60)
    print()
    
    packages_to_install = []
    
    # Check current status
    print("📋 Checking current installation status...")
    
    # Check EasyOCR
    if check_package('easyocr'):
        print("✅ EasyOCR is already installed")
    else:
        print("❌ EasyOCR not found")
        packages_to_install.append(('easyocr', 'EasyOCR (Recommended for historical maps)'))
    
    # Check Tesseract
    if check_package('pytesseract'):
        print("✅ PyTesseract is already installed")
    else:
        print("❌ PyTesseract not found")
        packages_to_install.append(('pytesseract', 'PyTesseract OCR'))
    
    # Check PIL
    if check_package('PIL'):
        print("✅ Pillow (PIL) is already installed")
    else:
        print("❌ Pillow not found")
        packages_to_install.append(('Pillow', 'Pillow (PIL) for image processing'))
    
    # Check OpenCV
    if check_package('cv2'):
        print("✅ OpenCV is already installed")
    else:
        print("❌ OpenCV not found")
        packages_to_install.append(('opencv-python', 'OpenCV for computer vision'))
    
    print()
    
    if not packages_to_install:
        print("🎉 All OCR dependencies are already installed!")
        print("You can now use real text recognition in MapScribe AI.")
        return
    
    print(f"📦 Found {len(packages_to_install)} packages to install:")
    for package, description in packages_to_install:
        print(f"   • {package} - {description}")
    
    print()
    response = input("Do you want to install these packages? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Installation cancelled.")
        return
    
    print()
    print("🚀 Starting installation...")
    print()
    
    # Install packages
    success_count = 0
    for package, description in packages_to_install:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully installed: {success_count}/{len(packages_to_install)} packages")
    
    if success_count == len(packages_to_install):
        print("🎉 All packages installed successfully!")
        print()
        print("🔧 Additional Setup Required:")
        print()
        
        # Check if Tesseract was installed
        if any('pytesseract' in pkg[0] for pkg in packages_to_install):
            print("📋 For Tesseract OCR, you also need to install the Tesseract binary:")
            print()
            if os.name == 'nt':  # Windows
                print("Windows:")
                print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
                print("2. Install the .exe file")
                print("3. Add Tesseract to your PATH or set TESSDATA_PREFIX")
            else:
                print("macOS: brew install tesseract")
                print("Ubuntu/Debian: sudo apt install tesseract-ocr")
                print("CentOS/RHEL: sudo yum install tesseract")
        
        print()
        print("🎯 Next Steps:")
        print("1. Restart your Python application")
        print("2. Test with: python real_image_processor.py")
        print("3. Run MapScribe AI: python mapscribe_app.py")
        
    else:
        failed_count = len(packages_to_install) - success_count
        print(f"⚠️  {failed_count} packages failed to install")
        print("You may need to install them manually or check your internet connection.")
    
    print()
    print("🗺️  Ready to use MapScribe AI with real text recognition!")

if __name__ == "__main__":
    main()