#!/usr/bin/env python3
"""
Quick Image Resizer - Make your large historical maps web-friendly
"""

import os
import sys
from PIL import Image

def quick_resize(image_path):
    """Quickly resize an image for web use."""
    try:
        print(f"🔄 Processing: {image_path}")
        
        with Image.open(image_path) as img:
            width, height = img.size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            
            print(f"📏 Original: {width}x{height} ({file_size_mb:.1f} MB)")
            
            # If already small enough, skip
            if max(width, height) <= 1500 and file_size_mb <= 5:
                print("✅ Image is already web-friendly!")
                return image_path
            
            # Resize to max 1500px
            max_size = 1500
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save with web-friendly name
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_web{ext}"
                
                resized.save(output_path, quality=85, optimize=True)
                
                new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✅ Resized: {new_width}x{new_height} ({new_size_mb:.1f} MB)")
                print(f"💾 Saved as: {output_path}")
                
                return output_path
            
            return image_path
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🖼️  Quick Image Resizer for Web Upload")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # Command line mode
        for image_path in sys.argv[1:]:
            if os.path.exists(image_path):
                result = quick_resize(image_path)
                if result:
                    print(f"🎉 Ready for web upload: {result}\n")
            else:
                print(f"❌ File not found: {image_path}\n")
    else:
        # Interactive mode
        while True:
            image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
                
            if os.path.exists(image_path):
                result = quick_resize(image_path)
                if result:
                    print(f"🎉 Ready for web upload: {result}")
            else:
                print(f"❌ File not found: {image_path}")
        
        print("👋 Goodbye!")