#!/bin/bash
# Render Build Script for MapScribe AI

echo "🚀 Starting MapScribe AI build process..."

# Install system dependencies for OpenCV
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements-render.txt

# Create necessary directories
mkdir -p static/uploads
mkdir -p static/results
mkdir -p static/demo

# Set permissions
chmod -R 755 static/

echo "✅ Build completed successfully!"