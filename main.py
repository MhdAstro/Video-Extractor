from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import tempfile
import os
from typing import List, Dict
import io
import logging
import time
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Frame Extractor API",
    description="API for extracting meaningful frames from videos using scene detection and motion analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def extract_frames_from_video(binary_data: bytes) -> List[Dict]:
    """
    Extract meaningful frames from video using scene detection and motion analysis.
    
    Args:
        binary_data (bytes): Binary video data
    Returns:
        List[Dict]: List of frames with metadata
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
        
        while True:
            if time.time() - start_time > 300:  # 5 minutes timeout
                logger.error("Processing timeout reached")
                raise HTTPException(status_code=408, detail="Processing timeout reached")
                
            ret, current_frame_bgr = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processing frame {frame_idx}/{total_frames_count}")
            
            gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            
            scene_score_val = 0.0
            motion_score_val = 0.0

            if prev_frame_gray is not None:
                frame_diff = cv2.absdiff(gray, prev_frame_gray)
                scene_score_val = np.mean(frame_diff)
                
                if scene_score_val <= min_scene_diff:
                    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, gray, None, 
                                                      0.5, 3, 15, 3, 5, 1.2, 0)
                    if flow is not None:
                         motion_score_val = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
                    else:
                         motion_score_val = 0.0

                should_save = False
                if scene_score_val > min_scene_diff:
                    should_save = True
                elif motion_score_val > min_motion_value:
                    should_save = True
                
                if should_save and (frame_idx - last_saved_frame_idx) > 15:
                    ret_encode, buffer = cv2.imencode('.jpg', current_frame_bgr)
                    if not ret_encode:
                        logger.warning(f"Failed to encode frame {frame_idx} to JPEG")
                        prev_frame_gray = gray
                        continue

                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    timestamp = frame_idx / fps
                    
                    frames_output.append({
                        'frameNumber': len(frames_output) + 1,
                        'timestamp': round(timestamp, 2),
                        'videoTotalFrames': total_frames_count,
                        'sceneScore': round(float(scene_score_val), 2),
                        'motionScore': round(float(motion_score_val), 2),
                        'imageData': img_base64
                    })
                    last_saved_frame_idx = frame_idx
            
            prev_frame_gray = gray
            
        cap.release()
        logger.info(f"Frame extraction completed. Extracted {len(frames_output)} frames")
        return frames_output
        
    except Exception as e:
        logger.error(f"Error during frame extraction: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")
        
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.info("Temporary file cleaned up")
            except Exception as e_unlink:
                logger.warning(f"Failed to delete temporary video file {video_path}: {str(e_unlink)}")

@app.post("/extract-frames/", tags=["Video Processing"])
async def extract_frames(video: UploadFile = File(...)):
    """
    Extract meaningful frames from an uploaded video file.
    """
    logger.info(f"Received video upload request - Filename: {video.filename}, Content-Type: {video.content_type}")
    
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        logger.warning(f"Unsupported file format: {video.filename}")
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video file.")
    
    try:
        start_time = time.time()
        logger.info("Reading uploaded file")
        contents = await video.read()
        file_size = len(contents) / (1024 * 1024)  # Size in MB
        logger.info(f"File size: {file_size:.2f} MB")
        
        if file_size > 100:  # 100MB limit
            logger.warning(f"File too large: {file_size:.2f} MB")
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
            
        frames = extract_frames_from_video(contents)
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return JSONResponse(
            content={"status": "success", "frames": frames},
            headers={"X-Processing-Time": str(processing_time)}
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Health Check"])
async def root():
    """
    Health check endpoint
    """
    logger.info("Health check request received")
    return {"status": "healthy", "message": "Video Frame Extractor API is running"} 