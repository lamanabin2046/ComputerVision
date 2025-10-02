# generate_chessboard.py
#
# Description:
# This script generates a 9x6 internal corner chessboard pattern and saves it
# as a high-resolution PNG file, sized to be printed on A4 paper.
# This version is corrected for compatibility with older OpenCV versions.

import cv2
import numpy as np

def create_a4_chessboard():
    """
    Generates a 9x6 chessboard centered on a 300 DPI A4 canvas.
    """
    # --- Configuration ---
    # A4 paper dimensions in pixels at 300 DPI
    # Dimension in pixels = (dimension in inches) * DPI
    # A4 dimensions: 8.27 x 11.69 inches

    A4_WIDTH_PX = 2480      # 8.27 inches * 300 DPI
    A4_HEIGHT_PX = 3508     # 11.69 inches * 300 DPI
    
    # Chessboard configuration (9x6 internal corners means 10x7 squares)
    BOARD_COLS = 10  # Number of squares horizontally
    BOARD_ROWS = 7   # Number of squares vertically
    SQUARE_SIZE_PX = 300 # Size of each square in pixels
    
    # Calculate total board dimensions
    board_w = BOARD_COLS * SQUARE_SIZE_PX
    board_h = BOARD_ROWS * SQUARE_SIZE_PX
    
    # Create a white A4-sized canvas (in landscape for the board)
    canvas = np.full((A4_WIDTH_PX, A4_HEIGHT_PX), 255, dtype=np.uint8)

    # Calculate top-left corner to center the board on the canvas
    start_x = (A4_HEIGHT_PX - board_w) // 2
    start_y = (A4_WIDTH_PX - board_h) // 2
    
    # --- Draw the Chessboard ---
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            # If the sum of row and col index is even, draw a black square
            if (r + c) % 2 == 0:
                top_left_x = start_x + (c * SQUARE_SIZE_PX)
                top_left_y = start_y + (r * SQUARE_SIZE_PX)
                bottom_right_x = top_left_x + SQUARE_SIZE_PX
                bottom_right_y = top_left_y + SQUARE_SIZE_PX
                
                # Draw the black rectangle
                cv2.rectangle(
                    canvas,
                    (top_left_x, top_left_y),
                    (bottom_right_x, bottom_right_y),
                    0, # Black color
                    -1 # Filled rectangle
                )
    
    # cv2.ROTATE_90_COUNTER_CLOCKWISE is equivalent to '2'.
    canvas_portrait = cv2.rotate(canvas, 2)

    # --- Save the File ---
    file_name = "A4_Chessboard_9x6.png"
    try:
        cv2.imwrite(file_name, canvas_portrait)
        print(f"Successfully generated '{file_name}'")
        print(f"Image dimensions: {canvas_portrait.shape[1]}x{canvas_portrait.shape[0]} pixels")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    create_a4_chessboard()
