from real_ocr_processor import RealOCRProcessor

processor = RealOCRProcessor()
result = processor.process_image('uploads/12684c39_20251003_002848_kirotestimage-2.jpg', {
    'task': 'task4', 
    'confidence_threshold': 0.1
})

if result['success']:
    texts = result['recognized_texts']
    print(f'\n🎉 Found {len(texts)} text regions in your image!')
    
    for i, text in enumerate(texts[:10], 1):  # Show first 10
        bbox = text['bbox']
        print(f'  {i}. "{text["text"]}" - confidence: {text["confidence"]:.3f}')
        print(f'     Location: ({bbox["x"]}, {bbox["y"]})')
else:
    print(f'❌ Error: {result["error"]}')