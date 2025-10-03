# How to Test Your Historical Map Images 🗺️

This guide shows you exactly how to test the historical map text processing system with your own images to detect and recognize place names.

## Quick Start 🚀

### Method 1: Simple Test Script (Recommended)

Use the simple test script for immediate results:

```bash
# Test a single image (best for place names)
python simple_image_test.py your_map.jpg --task task4

# Save results to file
python simple_image_test.py your_map.jpg --task task4 --output results.json

# Test with different tasks
python simple_image_test.py your_map.jpg --task task3  # Individual words
python simple_image_test.py your_map.jpg --task task2  # Detection only
```

### Method 2: Using the Demo Image

We created a demo image for you to test with:

```bash
# Test the demo image
python simple_image_test.py demo_historical_map.jpg --task task4
```

## Understanding the Tasks 📋

Choose the right task for your needs:

| Task | Description | Best For | Output |
|------|-------------|----------|---------|
| **task1** | Word Detection only | Finding individual words | Bounding boxes only |
| **task2** | Phrase Detection only | Finding text phrases | Bounding boxes only |
| **task3** | Word Detection + Recognition | Reading individual words | Boxes + recognized text |
| **task4** | Phrase Detection + Recognition | **Reading place names** | Boxes + recognized phrases |

**💡 Recommendation: Use `task4` for the best place name detection results!**

## What You'll Get 📊

When you process an image, you'll see:

### 1. Detection Results
- Number of text regions found
- Location coordinates (x, y)
- Size of each text region (width, height)
- Confidence scores (0.0 to 1.0)

### 2. Recognition Results (tasks 3 & 4)
- Actual text content (place names)
- Language detected
- Recognition confidence
- Exact location on the map

### 3. Spatial Linking (tasks 3 & 4)
- Map coordinates for each place
- Spatial confidence scores
- Geographic type classification

## Example Output 📍

```
📍 RECOGNIZED PLACES: 4 places identified
   1. 'Greater London' (en)
      Confidence: 0.86
      Location: (100, 150)
      Size: 120x25
   2. 'Westminster' (en)
      Confidence: 0.90
      Location: (200, 250)
      Size: 95x22
   3. 'Thames River' (en)
      Confidence: 0.88
      Location: (300, 180)
      Size: 110x24
   4. 'Tower Bridge' (en)
      Confidence: 0.82
      Location: (150, 350)
      Size: 140x20
```

## Advanced Usage 🔧

### Batch Processing Multiple Images

```bash
# Process all images in a folder
for image in images/*.jpg; do
    python simple_image_test.py "$image" --task task4 --output "results/$(basename "$image" .jpg).json"
done
```

### Using the API (When Available)

```bash
# Start the API server
python api_server.py

# Upload image via web interface
# Go to http://localhost:8000 in your browser
```

### Direct Python Integration

```python
# In your Python code
from simple_image_test import test_image_processing

# Process an image
results = test_image_processing("my_map.jpg", "task4")

# Extract place names
places = [text['text'] for text in results['recognized_texts']]
print("Found places:", places)
```

## File Formats Supported 📁

The system supports common image formats:
- **JPEG** (.jpg, .jpeg) - Recommended
- **PNG** (.png) - Good quality
- **TIFF** (.tiff, .tif) - High quality scans
- **BMP** (.bmp) - Uncompressed

## Tips for Best Results 💡

### Image Quality
- **Resolution**: Higher resolution images (1000+ pixels) work better
- **Contrast**: Good contrast between text and background
- **Clarity**: Sharp, non-blurry text is easier to recognize

### Historical Maps
- **Scan Quality**: Clean scans without artifacts
- **Text Size**: Text should be readable by human eye
- **Language**: Currently optimized for English and Chinese

### Task Selection
- **Place Names**: Use `task4` for multi-word place names
- **Individual Words**: Use `task3` for single words
- **Just Detection**: Use `task1` or `task2` if you only need locations

## Troubleshooting 🔧

### Common Issues

**"Image not found"**
- Check the file path is correct
- Make sure the file exists
- Use forward slashes (/) in paths

**"Could not load image"**
- Check the image format is supported
- Try converting to JPEG format
- Ensure the image file isn't corrupted

**"Low confidence scores"**
- Try preprocessing the image (enhance contrast)
- Check if text is clear and readable
- Consider using a higher resolution image

**"No places detected"**
- Make sure there's actually text in the image
- Try different tasks (task1-task4)
- Check if the text is in a supported language

### Getting Help

If you encounter issues:
1. Check the image quality and format
2. Try the demo image first to verify the system works
3. Experiment with different tasks
4. Look at the confidence scores to understand results

## Configuration Options ⚙️

You can adjust settings in `config/default_config.yaml`:

```yaml
detection:
  confidence_threshold: 0.5  # Lower = more detections
  nms_threshold: 0.4        # Overlap handling

recognition:
  languages: ["en", "zh"]   # Supported languages
  beam_width: 5            # Recognition accuracy vs speed

preprocessing:
  enhance_contrast: true    # Improve text visibility
  denoise: true            # Remove image noise
```

## Next Steps 🎯

After testing your images:

1. **Analyze Results**: Review confidence scores and accuracy
2. **Batch Process**: Process multiple images at once
3. **Export Data**: Save results to JSON/CSV for analysis
4. **Integration**: Use the API for web applications
5. **Optimization**: Adjust settings for better results

## Example Commands 📝

```bash
# Basic usage
python simple_image_test.py my_historical_map.jpg

# Best for place names
python simple_image_test.py my_map.jpg --task task4

# Save detailed results
python simple_image_test.py my_map.jpg --task task4 --output my_results.json

# Verbose output
python simple_image_test.py my_map.jpg --task task4 --verbose

# Test the demo
python simple_image_test.py demo_historical_map.jpg --task task4
```

## Success! 🎉

You should now be able to:
- ✅ Process your historical map images
- ✅ Detect text regions on maps
- ✅ Recognize place names and locations
- ✅ Get coordinate information for each place
- ✅ Save results for further analysis

Happy mapping! 🗺️✨