# ---------------------------------------------Region Extraction Service-------------------------------------------- #

import cv2
import numpy as np
from scipy.spatial import ConvexHull
import traceback

from app.core.config import settings
from app.core.logging import logger

class RegionExtractor:
    """Extract regions of interest from facial landmarks."""
    
    @staticmethod
    def extract_regions(image, landmarks):
        """
        Extract facial regions based on anatomical areas (T-zone, left cheek, right cheek).
        
        Args:
            image (numpy.ndarray): BGR image
            landmarks (list): List of (x, y) coordinates of facial landmarks
                
        Returns:
            dict: Dictionary containing region masks and bounding boxes
        """
        result = {
            "success": False,
            "regions": {
                "t_zone": {"points": [], "bounding_box": None},
                "left_cheek": {"points": [], "bounding_box": None},
                "right_cheek": {"points": [], "bounding_box": None}
            },
            "region_masks": {},
            "message": ""
        }
        
        try:
            h, w = image.shape[:2]
            
            # 1. Create T-zone shape directly using anatomical understanding
            t_zone_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Forehead - top of T
            eyebrow_points = [landmarks[i] for i in range(17, 27)]
            eyebrow_top_y = min(p[1] for p in eyebrow_points)
            eyebrow_left_x = landmarks[17][0]
            eyebrow_right_x = landmarks[26][0]
            
            # Extrapolate forehead height
            chin_y = landmarks[8][1]
            face_height = chin_y - eyebrow_top_y
            forehead_height = int(face_height * 0.3)  # 30% of face height for forehead
            forehead_top_y = max(0, eyebrow_top_y - forehead_height)
            
            # Draw forehead (top of T)
            forehead_points = np.array([
                [eyebrow_left_x, eyebrow_top_y],
                [eyebrow_left_x, forehead_top_y],
                [eyebrow_right_x, forehead_top_y],
                [eyebrow_right_x, eyebrow_top_y]
            ], dtype=np.int32)
            cv2.fillPoly(t_zone_mask, [forehead_points], 255)
            
            # Nose bridge to tip (middle of T)
            nose_width = int((eyebrow_right_x - eyebrow_left_x) * 0.3)  # 30% of face width
            nose_top_y = eyebrow_top_y
            nose_bottom_y = landmarks[33][1]
            nose_center_x = landmarks[30][0]
            
            nose_points = np.array([
                [nose_center_x - nose_width//2, nose_top_y],
                [nose_center_x + nose_width//2, nose_top_y],
                [nose_center_x + nose_width//2, nose_bottom_y],
                [nose_center_x - nose_width//2, nose_bottom_y]
            ], dtype=np.int32)
            cv2.fillPoly(t_zone_mask, [nose_points], 255)
            
            # Chin area (bottom of T)
            chin_width = nose_width
            chin_points = np.array([
                [nose_center_x - chin_width//2, nose_bottom_y],
                [nose_center_x + chin_width//2, nose_bottom_y],
                [landmarks[8][0] + chin_width//4, landmarks[8][1]],
                [landmarks[8][0] - chin_width//4, landmarks[8][1]]
            ], dtype=np.int32)
            cv2.fillPoly(t_zone_mask, [chin_points], 255)
            
            # 2. Create left cheek region
            left_cheek_mask = np.zeros((h, w), dtype=np.uint8)
                    
            left_cheek_points = [
                landmarks[0],  # Jawline start
                landmarks[1],  # Jawline
                landmarks[2],  # Jawline
                landmarks[3],  # Jawline
                landmarks[4],  # Jawline
                landmarks[5],  # Jawline
                landmarks[6],  # Jawline
                landmarks[7],  # Jawline
                landmarks[31],  # Side of nose 
                landmarks[48],  # Mouth corner
                landmarks[49],  # Upper lip
                landmarks[50],  # Upper lip
                # Under-eye points instead of eyebrow
                landmarks[41],  # Eye corner
                landmarks[40],  # Under eye
                landmarks[39],  # Under eye
                landmarks[36]   # Eye corner
            ]
            cv2.fillConvexPoly(left_cheek_mask, np.array(left_cheek_points, dtype=np.int32), 255)

            # 3. Create right cheek region
            right_cheek_mask = np.zeros((h, w), dtype=np.uint8)
                    
            right_cheek_points = [
                landmarks[16],  # Jawline start
                landmarks[15],  # Jawline
                landmarks[14],  # Jawline
                landmarks[13],  # Jawline
                landmarks[12],  # Jawline
                landmarks[11],  # Jawline
                landmarks[10],  # Jawline
                landmarks[9],   # Jawline
                landmarks[35],  # Side of nose
                landmarks[54],  # Mouth corner
                landmarks[53],  # Upper lip
                landmarks[52],  # Upper lip
                # Under-eye points instead of eyebrow
                landmarks[47],  # Eye corner
                landmarks[46],  # Under eye
                landmarks[45],  # Under eye
                landmarks[42]   # Eye corner
            ]
            cv2.fillConvexPoly(right_cheek_mask, np.array(right_cheek_points, dtype=np.int32), 255)
            
            # 4. Ensure regions don't overlap
            # Remove T-zone from cheeks
            left_cheek_mask = cv2.bitwise_and(left_cheek_mask, cv2.bitwise_not(t_zone_mask))
            right_cheek_mask = cv2.bitwise_and(right_cheek_mask, cv2.bitwise_not(t_zone_mask))
            
            # Store masks
            result["region_masks"]["t_zone"] = t_zone_mask
            result["region_masks"]["left_cheek"] = left_cheek_mask
            result["region_masks"]["right_cheek"] = right_cheek_mask
            
            # 5. Extract points and bounding boxes from masks
            for region_name, mask in result["region_masks"].items():
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(contour)
                    hull_points = hull.reshape(-1, 2)
                    
                    # Store points
                    result["regions"][region_name]["points"] = hull_points.tolist()
                    
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(hull)
                    result["regions"][region_name]["bounding_box"] = {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
            
            result["success"] = True
            result["message"] = "Regions extracted successfully"
            
        except Exception as e:
            logger.error(f"Error in region extraction: {str(e)}")
            logger.error(traceback.format_exc())
            result["message"] = f"Error extracting regions: {str(e)}"
        
        return result
    
    @staticmethod
    def extrapolate_forehead(landmarks, image_shape):
        """
        Create anatomically correct forehead region above eyebrows.
        
        Args:
            landmarks (list): List of (x, y) coordinates of facial landmarks
            image_shape (tuple): Image shape (height, width)
            
        Returns:
            list: List of forehead points
        """
        height, width = image_shape[:2]
        
        # Get eyebrow points (landmarks 17-26)
        if len(landmarks) < 27:
            return []
            
        left_eyebrow = landmarks[17:22]  # Left eyebrow
        right_eyebrow = landmarks[22:27]  # Right eyebrow
        
        # Calculate forehead top Y coordinate
        eyebrow_y_min = min(p[1] for p in left_eyebrow + right_eyebrow)
        chin_y = landmarks[8][1]  # Bottom of chin
        face_height = chin_y - eyebrow_y_min
        forehead_height = face_height * 0.35  # 35% of face height
        
        forehead_top_y = max(0, eyebrow_y_min - forehead_height)
        
        # Get temples (sides of forehead)
        left_temple_x = min(landmarks[0][0], left_eyebrow[0][0]) - width * 0.02  # Slightly wider
        right_temple_x = max(landmarks[16][0], right_eyebrow[-1][0]) + width * 0.02
        
        # Create a more natural forehead curve
        forehead_points = []
        
        # Left corner of forehead
        forehead_points.append([left_temple_x, eyebrow_y_min])
        
        # Left temple up
        forehead_points.append([left_temple_x, forehead_top_y + face_height*0.1])  # Slight curve
        
        # Top of forehead (several points for a natural curve)
        num_points = 7
        for i in range(num_points):
            x = left_temple_x + (right_temple_x - left_temple_x) * i / (num_points - 1)
            y = forehead_top_y
            forehead_points.append([x, y])
        
        # Right temple up
        forehead_points.append([right_temple_x, forehead_top_y + face_height*0.1])  # Slight curve
        
        # Right corner of forehead
        forehead_points.append([right_temple_x, eyebrow_y_min])
        
        return forehead_points
    
    @staticmethod
    def overlay_regions(image, regions, alpha=0.3):
        """
        Overlay region masks on the input image with different colors.
        
        Args:
            image (numpy.ndarray): BGR image
            regions (dict): Dictionary containing region masks
            alpha (float): Transparency factor (0-1)
            
        Returns:
            numpy.ndarray: Image with overlaid regions
        """
        # Colors for each region (BGR format)
        colors = {
            "t_zone": (0, 255, 0),    # Green
            "left_cheek": (0, 0, 255), # Red
            "right_cheek": (255, 0, 0) # Blue
        }
        
        visualization = image.copy()
        
        # Overlay each region
        for region_name, mask in regions.items():
            if mask is None:
                continue
            
            # Create colored overlay
            overlay = np.zeros_like(image)
            overlay[mask > 0] = colors[region_name]
            
            # Blend overlay with original image
            cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
            
            # Draw region contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visualization, contours, -1, colors[region_name], 2)
        
        # Add debugging text to show regions are being processed
        cv2.putText(visualization, "T-zone", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(visualization, "Left cheek", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(visualization, "Right cheek", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return visualization

# Create singleton instance
region_extractor = RegionExtractor()

# ---------------------------------------------------------------------------------------------------- #