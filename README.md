# Video Frame Extractor API

This API service extracts meaningful frames from videos using scene detection and motion analysis.

## Features

- Upload video files (MP4, AVI, MOV, MKV)
- Extract meaningful frames based on scene changes and motion
- Returns frames with metadata including timestamps and scores
- RESTful API with FastAPI
- Swagger documentation included

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`
Swagger documentation at `http://localhost:8000/docs`

## Deployment on Liara

1. Install Liara CLI:
```bash
npm install -g @liara/cli
```

2. Login to Liara:
```bash
liara login
```

3. Initialize the project (if not already done):
```bash
liara init
```

4. Deploy the application:
```bash
liara deploy
```

## API Endpoints

### POST /extract-frames/
Upload a video file to extract meaningful frames.

Example using curl:
```bash
curl -X POST "http://your-api-url/extract-frames/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your-video.mp4"
```

### GET /
Health check endpoint to verify the API is running.

## Response Format

The API returns JSON responses in the following format:

```json
{
  "status": "success",
  "frames": [
    {
      "frameNumber": 1,
      "timestamp": 0.5,
      "videoTotalFrames": 300,
      "sceneScore": 45.2,
      "motionScore": 1200.5,
      "imageData": "base64_encoded_image_data"
    }
  ]
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (invalid file format)
- 500: Internal Server Error (processing failed)

## Environment Variables

None required for basic operation.

## License

MIT 