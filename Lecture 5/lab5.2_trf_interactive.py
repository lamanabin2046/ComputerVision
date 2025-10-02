import cv2
import numpy as np
import requests
import os
from typing import Tuple, Dict, Optional

# --- CONSTANTS ---
IMAGE_URL = 'https://images.unsplash.com/photo-1599420186946-7b6fb4e297f0?q=80&w=1887&auto=format&fit=crop'
IMAGE_FILENAME = 'transformation_sample.jpg'

# Main Window Name
WINDOW_NAME = 'Transformation Visualizer'

# Trackbar configurations: (name, min_val, max_val, default_pos)
# Note: For trackbars where the 'zero' is not at the start, the default value is adjusted
# to be the trackbar's initial position.
TRACKBARS_CONFIG = {
    'Angle': ('Angle', -180, 180, 180), # Default 0, so position is 180
    'Translate X': ('Translate X', -150, 150, 150), # Default 0, so position is 150
    'Translate Y': ('Translate Y', -100, 100, 100), # Default 0, so position is 100
    'Scale': ('Scale', 20, 200, 100), # Represents 0.2 to 2.0, default 1.0
}

# --- HELPER FUNCTIONS ---
def download_image(url: str, fallback_name: str) -> None:
    """Downloads an image from a URL and saves it locally."""
    print(f"Attempting to download image from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(fallback_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded successfully as {fallback_name}")
    except requests.exceptions.RequestException as e:
        print(f"Could not download image: {e}. Will attempt to use or create a local file.")

def get_image(filename: str) -> Optional[np.ndarray]:
    """Loads an image from a file, creating a placeholder if it doesn't exist."""
    if not os.path.exists(filename):
        download_image(IMAGE_URL, filename)

    image = cv2.imread(filename)
    if image is None:
        print(f"Failed to load '{filename}'. Creating a placeholder.")
        img_placeholder = np.zeros((400, 600, 3), np.uint8)
        img_placeholder[:] = (47, 53, 66)  # Dark background
        text = "Sample Image"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 2, 3)
        text_x = (img_placeholder.shape[1] - text_width) // 2
        text_y = (img_placeholder.shape[0] + text_height) // 2
        cv2.putText(img_placeholder, text, (text_x, text_y), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(filename, img_placeholder)
        image = cv2.imread(filename)
    return image

def setup_ui(window_name: str) -> None:
    """Creates a single window and all associated trackbars."""
    cv2.namedWindow(window_name)
    dummy_callback = lambda x: None     # OpenCV trackbars require a callback function (called when the slider moves).
    
    for key, config in TRACKBARS_CONFIG.items():
        name, min_val, max_val, default_pos = config
        # Use max_val - min_val for the trackbar range if min_val is not 0
        range_max = max_val if min_val == 0 else max_val - min_val
        cv2.createTrackbar(name, window_name, default_pos, range_max, dummy_callback)

def main_loop(image: np.ndarray) -> None:
    """Runs the main application loop for transforming and displaying the image."""
    rows, cols, _ = image.shape
    center = (cols / 2, rows / 2)
    
    # Mode state: 0 for Rigid, 1 for Similarity.
    mode = 0 
    
    print("\nAdjust sliders. Press 'm' to switch modes. Press 'ESC' to quit.")
    
    while True:
        # --- Get parameters from trackbars ---
        angle_pos = cv2.getTrackbarPos(TRACKBARS_CONFIG['Angle'][0], WINDOW_NAME)
        tx_pos = cv2.getTrackbarPos(TRACKBARS_CONFIG['Translate X'][0], WINDOW_NAME)
        ty_pos = cv2.getTrackbarPos(TRACKBARS_CONFIG['Translate Y'][0], WINDOW_NAME)
        scale_pos = cv2.getTrackbarPos(TRACKBARS_CONFIG['Scale'][0], WINDOW_NAME)
        
        # Convert positions to actual values using min_val from config
        angle = angle_pos + TRACKBARS_CONFIG['Angle'][1]
        tx = tx_pos + TRACKBARS_CONFIG['Translate X'][1]
        ty = ty_pos + TRACKBARS_CONFIG['Translate Y'][1]
        
        # Determine scale based on the mode
        if mode == 0:  # Rigid Mode
            scale = 1.0
            mode_text = "Mode: Rigid (Press 'm' to switch)"
        else:  # Similarity Mode
            scale = scale_pos / 100.0
            mode_text = f"Mode: Similarity (Scale: {scale:.2f}x) (Press 'm' to switch)"

        # --- Apply Transformation ---
        transform_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        transform_matrix[0, 2] += tx
        transform_matrix[1, 2] += ty
        
        transformed_image = cv2.warpAffine(
            image, transform_matrix, (cols, rows), 
            borderMode=cv2.BORDER_CONSTANT, borderValue=(47, 53, 66)
        )
        
        # --- Add UI Text ---
        cv2.putText(
            transformed_image, mode_text, (40, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # --- Display Result ---
        cv2.imshow(WINDOW_NAME, transformed_image)

        # --- Handle Key Presses ---
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # 'ESC' key to exit
            break
        elif key == ord('m'): # 'm' key to switch mode
            mode = 1 - mode # Toggle between 0 and 1

def main():
    """Main function to set up and run the application."""
    image = get_image(IMAGE_FILENAME)
    if image is None:
        print("Could not load or create a sample image. Exiting.")
        return

    setup_ui(WINDOW_NAME)
    main_loop(image)
    
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()
