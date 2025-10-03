import sys
sys.path.insert(0, '.')

print("Testing direct import...")

try:
    import src.models.model_manager as mm
    print("Module imported successfully")
    print(f"Module file: {mm.__file__}")
    print(f"Module attributes: {dir(mm)}")
    
    # Try to access the classes
    if hasattr(mm, 'ModelManager'):
        print("ModelManager found!")
    else:
        print("ModelManager not found")
        
    if hasattr(mm, 'ModelVersion'):
        print("ModelVersion found!")
    else:
        print("ModelVersion not found")
        
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()