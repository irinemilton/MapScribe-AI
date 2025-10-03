import sys

# Test each import individually
try:
    print("Testing basic imports...")
    import os, json, logging, hashlib
    from pathlib import Path
    from typing import Dict, Any, Optional, List
    from datetime import datetime
    from dataclasses import dataclass, asdict
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")
    sys.exit(1)

try:
    print("Testing data structures import...")
    from src.models.data_structures import ModelMetadata, DetectionConfig, RecognitionConfig
    print("✓ Data structures import successful")
except Exception as e:
    print(f"✗ Data structures import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Testing detection interface import...")
    from src.detection.interfaces import DetectionModelInterface
    print("✓ Detection interface import successful")
except Exception as e:
    print(f"✗ Detection interface import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Testing recognition interface import...")
    from src.recognition.interfaces import RecognitionModelInterface
    print("✓ Recognition interface import successful")
except Exception as e:
    print(f"✗ Recognition interface import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All dependencies imported successfully!")

# Now test the model manager file execution
try:
    print("Testing model manager execution...")
    with open('src/models/model_manager.py', 'r') as f:
        code = f.read()
    
    # Create a namespace with all the imports
    namespace = {
        'os': os,
        'json': json,
        'logging': logging,
        'hashlib': hashlib,
        'Path': Path,
        'Dict': Dict,
        'Any': Any,
        'Optional': Optional,
        'List': List,
        'datetime': datetime,
        'dataclass': dataclass,
        'asdict': asdict,
        'ModelMetadata': ModelMetadata,
        'DetectionConfig': DetectionConfig,
        'RecognitionConfig': RecognitionConfig,
        'DetectionModelInterface': DetectionModelInterface,
        'RecognitionModelInterface': RecognitionModelInterface,
    }
    
    exec(code, namespace)
    print("✓ Model manager code executed successfully")
    
    if 'ModelManager' in namespace:
        print("✓ ModelManager class found in namespace")
    else:
        print("✗ ModelManager class not found in namespace")
        print(f"Available names: {[k for k in namespace.keys() if not k.startswith('_')]}")
        
except Exception as e:
    print(f"✗ Model manager execution failed: {e}")
    import traceback
    traceback.print_exc()