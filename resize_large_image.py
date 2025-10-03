#!/usr/bin/env python3
"""
Resize Large Images for Web Processing
Pre-process very large historical maps to make them web-friendly
"""

import os
import sys
from PIL import Image
import argparse

def resize_image(input_path, output_path=None, max_size=2048, quality=90):
    """Resize an image to make it web-friendly."""
    try:
        print(f"📁 Loading image: {input_path}")
        
        with Image.open(input_path) as img:
            original_width, original_height = img.size
            original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            
            print(f"📏 Original size: {original_width}x{original_height} ({original_size_mb:.2f} MB)")
            
            # Calculate new size
            max_dim = max(original_width, original_height)
            
            if max_dim <= max_size:
                print("✅ Image is already web-friendly size!")
                return input_path
            
            # Calculate scale factor
            scale = max_size / max_dim
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            print(f"🔄 Resizing to: {new_width}x{new_height} (scale: {scale:.3f})")
            
            # Resize with high quality
            resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Generate output path if not provided
            if output_path is None:
                name, ext = os.path.splitext(input_path)
                output_path = f"{name}_web_friendly{ext}"
            
            # Save with optimization
            resized.save(output_path, quality=quality, optimize=True)
            
            # Check new file size
            new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"✅ Saved resized image: {output_path}")
            print(f"📊 New size: {new_width}x{new_height} ({new_size_mb:.2f} MB)")
            print(f"💾 Size reduction: {((original_size_mb - new_size_mb) / original_size_mb * 100):.1f}%")
            
            return output_path
            
    except Exception as e:
        print(f"❌ Error resizing image: {e}")
        return None

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Resize large historical maps for web processing')
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output image file path (optional)')
    parser.add_argument('-s', '--size', type=int, default=2048, help='Maximum dimension (default: 2048)')
    parser.add_argument('-q', '--quality', type=int, default=90, help='JPEG quality (default: 90)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    print("🖼️  Large Image Resizer")
    print("=" * 30)
    
    result = resize_image(args.input, args.output, args.size, args.quality)
    
    if result:
        print(f"\n🎉 Success! Use this file in the web interface: {result}")
    else:
        print("\n❌ Failed to resize image.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode
        print("🖼️  Large Image Resizer")
        print("=" * 30)
        
        input_file = input("Enter image file path: ").strip()
        
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            sys.exit(1)
        
        result = resize_image(input_file)
        
        if result:
            print(f"\n🎉 Success! Use this file in the web interface: {result}")
        else:
            print("\n❌ Failed to resize image.")
    else:
        main()