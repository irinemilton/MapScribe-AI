# Historical Map Text Processing REST API Documentation

## Overview

The Historical Map Text Processing REST API provides programmatic access to all four ICDAR 2025 competition tasks through HTTP endpoints. The API supports file upload, batch processing, result retrieval, and system configuration.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, the API does not require authentication. This may be added in future versions.

## Content Types

- **Request**: `multipart/form-data` for file uploads, `application/json` for configuration
- **Response**: `application/json`

## Rate Limiting

The API implements basic rate limiting to prevent abuse. Current limits:
- 100 requests per minute per IP
- Maximum 10 files per batch request
- Maximum file size: 16MB

## Endpoints

### Health Check

Check API health and status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "detection": "available",
    "recognition": "available",
    "linking": "available",
    "evaluation": "available"
  }
}
```

### System Information

Get system and API information.

```http
GET /info
```

**Response:**
```json
{
  "system": {
    "platform": "Linux-5.4.0-x86_64",
    "python_version": "3.9.7",
    "pytorch_version": "1.12.0",
    "cuda_available": true,
    "cuda_devices": 1
  },
  "api": {
    "version": "1.0.0",
    "max_file_size": 16777216,
    "allowed_extensions": ["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
    "max_batch_size": 10
  },
  "tasks": {
    "task1": "Word Detection",
    "task2": "Phrase Detection",
    "task3": "Word Detection + Recognition",
    "task4": "Phrase Detection + Recognition"
  }
}
```

### Process Single Image

Process a single image for the specified competition task.

```http
POST /process/{task}
```

**Parameters:**
- `task` (path): Competition task (`task1`, `task2`, `task3`, `task4`)

**Form Data:**
- `file` (file): Image file to process
- `confidence` (float, optional): Confidence threshold (0.0-1.0, default: 0.5)
- `format` (string, optional): Output format (`txt`, `json`, default: `txt`)
- `save_intermediate` (boolean, optional): Save intermediate results (default: false)

**Example Request:**
```bash
curl -X POST \
  http://localhost:5000/process/task1 \
  -F "file=@historical_map.jpg" \
  -F "confidence=0.7" \
  -F "format=json"
```

**Response:**
```json
{
  "result_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "task1",
  "task_name": "Word Detection",
  "filename": "historical_map.jpg",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "processing_time": 2.34,
  "results": {
    "detections": [
      {
        "bbox": {"x": 100, "y": 50, "width": 200, "height": 30},
        "confidence": 0.85,
        "type": "word",
        "coordinates": [100, 50, 300, 80]
      }
    ],
    "statistics": {
      "detection_count": 5,
      "recognition_count": 0,
      "average_confidence": 0.82
    }
  },
  "download_url": "/results/550e8400-e29b-41d4-a716-446655440000/download",
  "result_url": "/results/550e8400-e29b-41d4-a716-446655440000"
}
```

### Batch Process Images

Process multiple images in a single request.

```http
POST /batch/{task}
```

**Parameters:**
- `task` (path): Competition task (`task1`, `task2`, `task3`, `task4`)

**Form Data:**
- `files` (file[]): Array of image files to process
- `confidence` (float, optional): Confidence threshold (default: 0.5)
- `format` (string, optional): Output format (default: `txt`)

**Example Request:**
```bash
curl -X POST \
  http://localhost:5000/batch/task3 \
  -F "files=@map1.jpg" \
  -F "files=@map2.png" \
  -F "confidence=0.6"
```

**Response:**
```json
{
  "batch_id": "batch-550e8400-e29b-41d4-a716-446655440000",
  "task": "task3",
  "task_name": "Word Detection + Recognition",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "summary": {
    "total_files": 2,
    "successful": 2,
    "failed": 0,
    "success_rate": 1.0,
    "total_processing_time": 4.67,
    "average_processing_time": 2.34
  },
  "statistics": {
    "total_detections": 12,
    "total_recognitions": 12,
    "average_detections_per_image": 6.0,
    "average_recognitions_per_image": 6.0
  },
  "results": [
    {
      "file_id": "batch-550e8400-e29b-41d4-a716-446655440000_0",
      "filename": "map1.jpg",
      "success": true,
      "processing_time": 2.1,
      "detection_count": 6,
      "recognition_count": 6,
      "download_url": "/results/batch-550e8400-e29b-41d4-a716-446655440000_0/download"
    }
  ]
}
```

### Get Processing Result

Retrieve information about a processing result.

```http
GET /results/{result_id}
```

**Parameters:**
- `result_id` (path): Result identifier

**Response:**
```json
{
  "result_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "task1",
  "created_at": "2024-01-15T10:30:00Z",
  "file_size": 1024,
  "line_count": 5,
  "download_url": "/results/550e8400-e29b-41d4-a716-446655440000/download",
  "status": "available"
}
```

### Download Result File

Download the result file for a processing result.

```http
GET /results/{result_id}/download
```

**Parameters:**
- `result_id` (path): Result identifier

**Response:**
- Content-Type: `text/plain` or `application/json`
- Content-Disposition: `attachment; filename="result_id_results.txt"`

### Configuration Management

#### Get Configuration

```http
GET /config
```

**Response:**
```json
{
  "configuration": {
    "detection": {
      "confidence_threshold": 0.5,
      "nms_threshold": 0.4,
      "model_path": "models/detection/yolo.pt"
    },
    "recognition": {
      "confidence_threshold": 0.3,
      "languages": ["en", "zh"]
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "success"
}
```

#### Update Configuration

```http
POST /config
Content-Type: application/json
```

**Request Body:**
```json
{
  "detection.confidence_threshold": 0.7,
  "recognition.languages": ["en", "zh", "fr"]
}
```

**Response:**
```json
{
  "message": "Configuration updated successfully",
  "updated_keys": ["detection.confidence_threshold", "recognition.languages"],
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "success"
}
```

### Model Management

#### List Available Models

```http
GET /models
```

**Response:**
```json
{
  "models": {
    "detection_models": [
      {
        "name": "YOLO Word Detector",
        "type": "yolo",
        "version": "1.0",
        "description": "YOLO-based word detection model"
      }
    ],
    "recognition_models": [
      {
        "name": "CRNN Recognizer",
        "type": "crnn",
        "version": "1.0",
        "description": "CRNN-based text recognition model"
      }
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "success"
}
```

#### Switch Model Version

```http
POST /models/{model_type}/switch
Content-Type: application/json
```

**Parameters:**
- `model_type` (path): Model type (`detection`, `recognition`)

**Request Body:**
```json
{
  "version": "2.0"
}
```

**Response:**
```json
{
  "message": "Switched detection model to version 2.0",
  "model_type": "detection",
  "version": "2.0"
}
```

### Evaluation

Evaluate results against ground truth data.

```http
POST /evaluate/{task}
Content-Type: application/json
```

**Parameters:**
- `task` (path): Competition task to evaluate

**Request Body:**
```json
{
  "results": [
    {
      "bbox": [100, 50, 300, 80],
      "confidence": 0.85,
      "text": "Sample Text"
    }
  ],
  "ground_truth": [
    {
      "bbox": [105, 52, 295, 78],
      "text": "Sample Text"
    }
  ]
}
```

**Response:**
```json
{
  "task": "task3",
  "task_name": "Word Detection + Recognition",
  "evaluation": {
    "metrics": {
      "precision": 0.92,
      "recall": 0.88,
      "f1_score": 0.90,
      "iou": 0.85,
      "accuracy": 0.89
    },
    "summary": {
      "total_predictions": 10,
      "total_ground_truth": 12,
      "evaluation_timestamp": "2024-01-15T10:30:00Z"
    }
  },
  "status": "completed"
}
```

## Error Responses

All error responses follow a consistent format:

```json
{
  "error": "Error Type",
  "message": "Human-readable error message",
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **413 Payload Too Large**: File size exceeds limit
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server processing error
- **503 Service Unavailable**: Service temporarily unavailable

### Validation Errors

Validation errors include detailed field-level error messages:

```json
{
  "error": "Validation Error",
  "message": "The request contains invalid parameters",
  "validation_errors": {
    "confidence": "Must be a number between 0 and 1",
    "format": "Must be one of: txt, json"
  },
  "status_code": 422,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Usage Examples

### Python Client Example

```python
import requests
import json

# Process single image
with open('historical_map.jpg', 'rb') as f:
    files = {'file': f}
    data = {'confidence': 0.7, 'format': 'json'}
    
    response = requests.post(
        'http://localhost:5000/process/task1',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Processing completed: {result['result_id']}")

# Download result
result_id = result['result_id']
download_response = requests.get(
    f'http://localhost:5000/results/{result_id}/download'
)

with open('result.txt', 'wb') as f:
    f.write(download_response.content)
```

### JavaScript Client Example

```javascript
// Process single image
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('confidence', '0.7');
formData.append('format', 'json');

fetch('http://localhost:5000/process/task1', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(result => {
    console.log('Processing completed:', result.result_id);
    
    // Download result
    return fetch(`http://localhost:5000/results/${result.result_id}/download`);
})
.then(response => response.text())
.then(resultText => {
    console.log('Result:', resultText);
});
```

### cURL Examples

```bash
# Process single image
curl -X POST \
  http://localhost:5000/process/task1 \
  -F "file=@map.jpg" \
  -F "confidence=0.7"

# Batch process
curl -X POST \
  http://localhost:5000/batch/task3 \
  -F "files=@map1.jpg" \
  -F "files=@map2.png"

# Get system info
curl http://localhost:5000/info

# Update configuration
curl -X POST \
  http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"detection.confidence_threshold": 0.8}'
```

## Development and Testing

### Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m src.api.main_api

# Run with custom configuration
python -c "
from src.api.main_api import HistoricalMapAPI
api = HistoricalMapAPI({'debug': True, 'port': 8000})
api.run()
"
```

### Testing Endpoints

Use the provided test scripts to validate API functionality:

```bash
# Run API tests
python -m pytest tests/test_api.py -v

# Test with sample data
python examples/api_examples.py
```

### API Client Libraries

Consider using these libraries for easier integration:

- **Python**: `requests`, `httpx`
- **JavaScript**: `fetch`, `axios`
- **Java**: `OkHttp`, `Apache HttpClient`
- **C#**: `HttpClient`

## Performance Considerations

- **File Size**: Keep uploaded files under 16MB for optimal performance
- **Batch Size**: Limit batch requests to 10 files maximum
- **Concurrent Requests**: API can handle multiple concurrent requests
- **Caching**: Results are cached temporarily for quick retrieval
- **Cleanup**: Old files are automatically cleaned up after 24 hours

## Security Notes

- Input validation is performed on all uploads
- File types are restricted to supported image formats
- Temporary files are stored securely and cleaned up automatically
- Consider implementing authentication for production use