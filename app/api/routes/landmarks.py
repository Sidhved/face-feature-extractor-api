# ---------------------------------------------Landmark API Routes-------------------------------------------- #

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import numpy as np
import uuid
import os
import time
from io import BytesIO

from app.api.models.request_models import ImageUrlRequest
from app.api.models.response_models import FacialFeatureResponse, ImageInfo, BoundingBox, Region
from app.services.landmark_detection import landmark_detector
from app.services.region_extraction import region_extractor
from app.services.image_utils import image_utils
from app.services.storage_service import storage_service
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

@router.post("/detect/upload", response_model=FacialFeatureResponse)
async def detect_landmarks_from_upload(
    file: UploadFile = File(...),
    save_result: bool = Query(None, description="Override default setting to save or not save the result JSON"),
    save_image: bool = Query(None, description="Override default setting to save or not save the processed image")
):
    """
    Detect facial landmarks from uploaded image file.
    
    - **file**: Image file to upload (JPEG, PNG)
    
    Returns facial landmarks and regions of interest.
    """
    start_time = time.time()
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    logger.info(f"Processing upload request {request_id}")
    
    # Check file extension
    filename = file.filename
    extension = os.path.splitext(filename)[1][1:].lower()
    if extension not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid file extension: {extension}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Invalid file format. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
                "face_detected": False,
                "processing_time": time.time() - start_time
            }
        )
    
    # Read file content
    contents = await file.read()
    if len(contents) > settings.MAX_IMAGE_SIZE:
        logger.warning(f"File too large: {len(contents)} bytes")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Image size exceeds the {settings.MAX_IMAGE_SIZE // (1024 * 1024)}MB limit",
                "face_detected": False,
                "processing_time": time.time() - start_time
            }
        )
    
    # Process image
    return await process_image(contents, filename, request_id, start_time, save_result, save_image)

@router.post("/detect/url", response_model=FacialFeatureResponse)
async def detect_landmarks_from_url(
    request: ImageUrlRequest,
    save_result: bool = Query(None, description="Override default setting to save or not save the result JSON"),
    save_image: bool = Query(None, description="Override default setting to save or not save the processed image")
):
    """
    Detect facial landmarks from image URL.
    
    - **url**: URL of the image to process
    
    Returns facial landmarks and regions of interest.
    """
    start_time = time.time()
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    logger.info(f"Processing URL request {request_id}: {request.url}")
    
    # Extract filename from URL
    url_path = str(request.url).split("/")[-1]
    filename = url_path.split("?")[0]  # Remove query parameters
    
    try:
        # Download image from URL
        image = image_utils.read_image_from_url(request.url)
        if image is None:
            logger.warning(f"Failed to download image from URL: {request.url}")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Failed to download or process image from URL",
                    "face_detected": False,
                    "processing_time": time.time() - start_time
                }
            )
        
        # Convert image to bytes for processing
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            raise Exception("Failed to encode image")
        
        image_bytes = buffer.tobytes()
        
        # Process image
        return await process_image(image_bytes, filename, request_id, start_time, save_result, save_image)
        
    except Exception as e:
        logger.error(f"Error processing URL {request.url}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error processing image: {str(e)}",
                "face_detected": False,
                "processing_time": time.time() - start_time
            }
        )

async def process_image(image_bytes, filename, request_id, start_time, save_result=None, save_image=None):
    """Process image and detect landmarks."""
    try:
        # Read image
        image = image_utils.read_image_from_bytes(image_bytes)
        if image is None:
            logger.warning(f"Failed to decode image for request {request_id}")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Failed to decode image. The file may be corrupted or in an unsupported format.",
                    "face_detected": False,
                    "processing_time": time.time() - start_time
                }
            )
        
        # Check image quality
        is_good, quality_message = image_utils.check_image_quality(image)
        if not is_good:
            logger.warning(f"Image quality check failed: {quality_message}")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": quality_message,
                    "face_detected": False,
                    "processing_time": time.time() - start_time
                }
            )
        
        # Prepare image info
        h, w = image.shape[:2]
        image_info = {
            "image_id": request_id,
            "filename": filename,
            "dimensions": {"width": w, "height": h},
            "format": os.path.splitext(filename)[1][1:].upper() if "." in filename else "UNKNOWN"
        }
        
        # Resize image for processing
        processed_image = image_utils.resize_image(image)
        
        # Detect landmarks
        landmark_result = landmark_detector.detect_landmarks(processed_image)
        
        if not landmark_result["success"]:
            logger.warning(f"No landmarks detected: {landmark_result['message']}")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": landmark_result["message"],
                    "image_info": image_info,
                    "face_detected": False,
                    "processing_time": time.time() - start_time
                }
            )
        
        # Extract regions of interest
        region_result = region_extractor.extract_regions(processed_image, landmark_result["landmarks"])
        
        # Generate visualization
        vis_img = landmark_detector.draw_landmarks(processed_image, landmark_result["landmarks"])
        
        if region_result["success"]:
            # Overlay regions on visualization
            vis_img = region_extractor.overlay_regions(
                vis_img, 
                region_result["region_masks"]
            )
        
        # Convert visualization to base64
        base64_image = image_utils.convert_to_base64(vis_img)
        
        # Prepare regions for response
        regions_dict = {}
        if region_result["success"]:
            for region_name, region_data in region_result["regions"].items():
                if region_data["points"] and region_data["bounding_box"]:
                    regions_dict[region_name] = region_data
        
        # Prepare response
        response_data = {
            "status": "success",
            "message": "Image processed successfully",
            "image_info": image_info,
            "face_detected": True,
            "landmarks": landmark_result["landmarks"],
            "regions_of_interest": regions_dict if regions_dict else None,
            "masked_image": base64_image,
            "processing_time": time.time() - start_time
        }
        
        # Save results if configured
        should_save_image = save_image if save_image is not None else settings.SAVE_IMAGES
        if should_save_image:
            save_success, file_path = storage_service.save_image(vis_img, image_info["image_id"])
            if save_success:
                logger.info(f"Visualization image saved to {file_path}")
                # Add the file path to the response
                response_data["visualization_file"] = file_path
            else:
                logger.warning("Failed to save visualization image")

        # Save processed image if configured
        should_save_result = save_result if save_result is not None else settings.SAVE_RESULTS
        if should_save_result:
            save_success, json_path, masked_image_path = storage_service.save_result(response_data)
            if save_success:
                logger.info(f"Result saved to {json_path}")
                # Add the file paths to the response
                response_data["result_file"] = json_path
                if masked_image_path:
                    response_data["masked_image_file"] = masked_image_path
            else:
                logger.warning("Failed to save result")
        
        logger.info(f"Successfully processed request {request_id} in {response_data['processing_time']:.2f} seconds")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing image for request {request_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}",
                "face_detected": False,
                "processing_time": time.time() - start_time
            }
        )
# ---------------------------------------------------------------------------------------------------- #