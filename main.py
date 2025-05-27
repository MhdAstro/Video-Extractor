from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import tempfile
import os
from typing import List, Dict
import io
import logging
import time
from fastapi.responses import JSONResponse, HTMLResponse
import asyncio
from starlette.requests import Request
from starlette.responses import Response
from fastapi.middleware.base import BaseHTTPMiddleware
import socket
import requests
from pydantic import BaseModel, HttpUrl
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=300.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout"}
            )

class VideoURL(BaseModel):
    url: HttpUrl

app = FastAPI(
    title="Video Frame Extractor API",
    description="API for extracting meaningful frames from videos using scene detection and motion analysis",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")

# Add middlewares
app.add_middleware(TimeoutMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Headers: {request.headers}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Video Frame Extractor API - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/api/health")
async def health_check():
    """
    Detailed health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "opencv_version": cv2.__version__,
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "extract_frames": "/extract-frames/"
        }
    }

@app.get("/", tags=["Health Check"])
async def root(request: Request):
    """
    Basic health check endpoint
    """
    logger.info("Health check request received")
    return {
        "status": "healthy",
        "message": "Video Frame Extractor API is running",
        "request_headers": dict(request.headers),
        "host": socket.gethostname()
    }

@app.get("/debug", tags=["Debug"])
async def debug_info(request: Request):
    """
    Debug endpoint to show request information
    """
    return {
        "headers": dict(request.headers),
        "client": request.client,
        "url": str(request.url),
        "base_url": str(request.base_url),
        "host": socket.gethostname(),
        "environment": {
            "TMPDIR": os.getenv("TMPDIR"),
            "PORT": os.getenv("PORT"),
            "HOST": os.getenv("HOST")
        }
    }

def extract_frames_from_video(binary_data: bytes) -> List[Dict]:
    """
    Extract meaningful frames from video using scene detection and motion analysis.
    """
    logger.info("Starting frame extraction process")
    start_time = time.time()
    
    # Create temporary file for video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        logger.info("Creating temporary file")
        temp_video.write(binary_data)
        video_path = temp_video.name
        logger.info(f"Temporary file created at: {video_path}")

    try:
        # Open video file
        logger.info("Opening video file with OpenCV")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cv2_error = cv2.getBuildInformation()
            logger.error(f"Failed to open video file: {cv2_error[:500]}")
            raise Exception(f"Could not open video file. OpenCV build info: {cv2_error[:500]}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logger.warning("Video FPS is 0. Defaulting to 30 for calculations.")
            fps = 30.0 
            
        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames_count}")
        
        frames_output = []
        prev_frame_gray = None
        frame_idx = 0
        min_scene_diff = 30.0
        min_motion_value = 1000.0
        last_saved_frame_idx = -15
        
        # Add batch processing
        batch_size = 10  # Process 10 frames at a time
        batch_start_time = time.time()
        
        while True:
            current_time = time.time()
            if current_time - start_time > 300:  # 5 minutes timeout
                logger.error("Processing timeout reached")
                raise HTTPException(status_code=408, detail="Processing timeout reached")
            
            # Process frames in batches
            frames_processed_in_batch = 0
            while frames_processed_in_batch < batch_size:
                ret, current_frame_bgr = cap.read()
                if not ret:
                    logger.info("Reached end of video")
                    break
                
                frame_idx += 1
                frames_processed_in_batch += 1
                
                if frame_idx % 10 == 0:  # Log every 10 frames
                    elapsed_time = time.time() - batch_start_time
                    fps_processing = 10 / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"Processing frame {frame_idx}/{total_frames_count} (Processing speed: {fps_processing:.2f} fps)")
                    batch_start_time = time.time()
                
                try:
                    # Convert to grayscale
                    gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
                    
                    scene_score_val = 0.0
                    motion_score_val = 0.0

                    if prev_frame_gray is not None:
                        # Calculate scene difference
                        frame_diff = cv2.absdiff(gray, prev_frame_gray)
                        scene_score_val = float(np.mean(frame_diff))
                        
                        # Calculate motion if needed
                        if scene_score_val <= min_scene_diff:
                            try:
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_frame_gray, gray, None,
                                    0.5, 3, 15, 3, 5, 1.2, 0
                                )
                                if flow is not None:
                                    motion_score_val = float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
                            except Exception as e:
                                logger.warning(f"Error calculating optical flow: {str(e)}")
                                motion_score_val = 0.0

                        # Determine if frame should be saved
                        should_save = (scene_score_val > min_scene_diff or 
                                     motion_score_val > min_motion_value) and \
                                    (frame_idx - last_saved_frame_idx) > 15

                        if should_save:
                            try:
                                ret_encode, buffer = cv2.imencode('.jpg', current_frame_bgr)
                                if not ret_encode:
                                    logger.warning(f"Failed to encode frame {frame_idx} to JPEG")
                                    continue

                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                timestamp = frame_idx / fps
                                
                                frames_output.append({
                                    'frameNumber': len(frames_output) + 1,
                                    'timestamp': round(timestamp, 2),
                                    'videoTotalFrames': total_frames_count,
                                    'sceneScore': round(scene_score_val, 2),
                                    'motionScore': round(motion_score_val, 2),
                                    'imageData': img_base64
                                })
                                last_saved_frame_idx = frame_idx
                                logger.info(f"Saved frame {frame_idx} (Scene score: {scene_score_val:.2f}, Motion score: {motion_score_val:.2f})")
                            except Exception as e:
                                logger.error(f"Error saving frame {frame_idx}: {str(e)}")
                                continue
                    
                    prev_frame_gray = gray
                
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                    continue
            
            if frames_processed_in_batch < batch_size:  # End of video
                break
        
        cap.release()
        logger.info(f"Frame extraction completed. Extracted {len(frames_output)} frames")
        return frames_output
        
    except Exception as e:
        logger.error(f"Error during frame extraction: {str(e)}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")
        
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.info("Temporary file cleaned up")
            except Exception as e_unlink:
                logger.warning(f"Failed to delete temporary video file {video_path}: {str(e_unlink)}")

async def download_video(url: str) -> bytes:
    """
    Download video from URL
    """
    logger.info(f"Downloading video from URL: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(video_type in content_type for video_type in ['video/', 'application/octet-stream']):
            raise HTTPException(status_code=400, detail=f"URL does not point to a video file. Content-Type: {content_type}")
        
        # Get file extension from URL or content-type
        file_extension = os.path.splitext(urlparse(url).path)[1].lower()
        if not file_extension:
            if 'mp4' in content_type:
                file_extension = '.mp4'
            elif 'avi' in content_type:
                file_extension = '.avi'
            elif 'mov' in content_type:
                file_extension = '.mov'
            elif 'mkv' in content_type:
                file_extension = '.mkv'
            else:
                file_extension = '.mp4'  # default to mp4
        
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv']:
            raise HTTPException(status_code=400, detail=f"Unsupported video format: {file_extension}")
        
        # Download the file
        video_data = response.content
        file_size = len(video_data) / (1024 * 1024)  # Size in MB
        logger.info(f"Downloaded video size: {file_size:.2f} MB")
        
        if file_size > 100:  # 100MB limit
            raise HTTPException(status_code=413, detail="Video file too large. Maximum size is 100MB")
            
        return video_data
        
    except requests.RequestException as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading video: {str(e)}")

@app.post("/extract-frames/", tags=["Video Processing"])
async def extract_frames(video_url: VideoURL):
    """
    Extract meaningful frames from a video URL.
    
    Parameters:
    - video_url: URL of the video file to process
    
    Returns:
    - List of frames with metadata
    """
    try:
        start_time = time.time()
        logger.info(f"Processing video from URL: {video_url.url}")
        
        # Download video
        video_data = await download_video(video_url.url)
        
        # Process frames
        frames = extract_frames_from_video(video_data)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return JSONResponse(
            content={"status": "success", "frames": frames},
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Frames-Extracted": str(len(frames))
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 