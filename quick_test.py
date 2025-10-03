from simple_processor import SimpleMapProcessor

processor = SimpleMapProcessor()
result = processor.process_image('demo_historical_map.jpg', {'task': 'task4'})

print(f'Found {len(result["recognized_texts"])} regions')
for i, text in enumerate(result['recognized_texts'][:5], 1):
    bbox = text['bbox']
    print(f'  {i}. {text["text"]} at ({bbox["x"]}, {bbox["y"]}) - confidence: {text["confidence"]:.2f}')