#!/usr/bin/env python3
"""
MapScribe AI - Setup Configuration
Historical Map Text Processing System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mapscribe-ai",
    version="1.0.0",
    author="MapScribe AI Team",
    author_email="contact@mapscribe-ai.com",
    description="Advanced AI-powered system for extracting and categorizing text from historical maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irinemilton/MapScribe-AI",
    project_urls={
        "Bug Tracker": "https://github.com/irinemilton/MapScribe-AI/issues",
        "Documentation": "https://github.com/irinemilton/MapScribe-AI/wiki",
        "Source Code": "https://github.com/irinemilton/MapScribe-AI",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "gpu": [
            "torch>=1.9.0+cu117",
            "torchvision>=0.10.0+cu117",
        ],
    },
    entry_points={
        "console_scripts": [
            "mapscribe=src.cli.main_cli:main",
            "mapscribe-web=start_web_server:main",
            "mapscribe-api=api_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mapscribe": [
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
            "config/*.yaml",
            "docs/*.md",
        ],
    },
    keywords=[
        "historical maps",
        "OCR",
        "text recognition",
        "computer vision",
        "digital humanities",
        "cartography",
        "machine learning",
        "image processing",
        "geographic information",
        "ICDAR competition",
    ],
    zip_safe=False,
)