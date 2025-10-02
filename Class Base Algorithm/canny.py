import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

class CannyEdgeDetector:
    def __init__(self, image_path, sigma=1, low_threshold=0.05, high_threshold=0.15):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Error: Could not read image from path '{image_path}'")
        self.image = image
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    # -----------------------------
    # Step 1: Gaussian Blur
    # -----------------------------
    def gaussian_blur(self):
        return ndimage.gaussian_filter(self.image, self.sigma)

    # -----------------------------
    # Step 2: Gradient Magnitude & Direction
    # -----------------------------
    def sobel_filters(self, img):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        Gx = ndimage.convolve(img, sobel_x)
        Gy = ndimage.convolve(img, sobel_y)
        magnitude = np.hypot(Gx, Gy)
        magnitude = (magnitude / magnitude.max()) * 255
        direction = np.arctan2(Gy, Gx) * 180 / np.pi
        direction[direction < 0] += 180
        return magnitude, direction

    # -----------------------------
    # Step 3: Non-Maximum Suppression (Vectorized)
    # -----------------------------
    def non_maximum_suppression(self, magnitude, direction):
        M, N = magnitude.shape
        Z = np.zeros((M, N), dtype=np.float32)
        angle = direction.copy()
        angle[angle < 0] += 180

        # Round angle to 4 main directions: 0, 45, 90, 135
        angle = (np.round(angle / 45) * 45) % 180

        # Pad the image to avoid boundary issues
        mag_padded = np.pad(magnitude, ((1,1),(1,1)), mode='constant')

        # Compare neighbors based on direction
        for i in range(1, M+1):
            for j in range(1, N+1):
                q = 255
                r = 255
                if angle[i-1, j-1] == 0:
                    q = mag_padded[i, j+1]
                    r = mag_padded[i, j-1]
                elif angle[i-1, j-1] == 45:
                    q = mag_padded[i-1, j+1]
                    r = mag_padded[i+1, j-1]
                elif angle[i-1, j-1] == 90:
                    q = mag_padded[i-1, j]
                    r = mag_padded[i+1, j]
                elif angle[i-1, j-1] == 135:
                    q = mag_padded[i-1, j-1]
                    r = mag_padded[i+1, j+1]
                if mag_padded[i, j] >= q and mag_padded[i, j] >= r:
                    Z[i-1, j-1] = mag_padded[i, j]
        return Z

    # -----------------------------
    # Step 4: Double Threshold
    # -----------------------------
    def double_threshold(self, img):
        high_val = img.max() * self.high_threshold
        low_val = img.max() * self.low_threshold
        strong_edges = np.zeros_like(img, dtype=np.uint8)
        weak_edges = np.zeros_like(img, dtype=np.uint8)
        strong_edges[img >= high_val] = 255
        weak_edges[(img >= low_val) & (img < high_val)] = 75
        return strong_edges, weak_edges

    # -----------------------------
    # Step 5: Edge Tracking by Hysteresis (Vectorized)
    # -----------------------------
    def hysteresis(self, strong_edges, weak_edges):
        M, N = strong_edges.shape
        final_edges = strong_edges.copy()
        # Identify weak pixels that have strong neighbors
        shift_coords = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        weak_y, weak_x = np.where(weak_edges == 75)
        for y, x in zip(weak_y, weak_x):
            for dy, dx in shift_coords:
                ny, nx = y+dy, x+dx
                if 0 <= ny < M and 0 <= nx < N:
                    if strong_edges[ny, nx] == 255:
                        final_edges[y, x] = 255
                        break
        return final_edges

    # -----------------------------
    # Full Pipeline
    # -----------------------------
    def process(self):
        smoothed = self.gaussian_blur()
        magnitude, direction = self.sobel_filters(smoothed)
        suppressed = self.non_maximum_suppression(magnitude, direction)
        strong, weak = self.double_threshold(suppressed)
        final_edges = self.hysteresis(strong, weak)
        return smoothed, magnitude, suppressed, strong, weak, final_edges

# -----------------------------
# Run Example
if __name__ == '__main__':
    import cv2
    import numpy as np

    image_filename = 'images/3d_drawings.jpg'

    try:
        # Initialize detector and process
        detector = CannyEdgeDetector(image_filename)
        smoothed, magnitude, suppressed, strong, weak, final_edges = detector.process()

        # Images and their titles
        images = [
            (detector.image, 'Original Image'),
            (smoothed, 'Gaussian Blur'),
            (magnitude, 'Gradient Magnitude'),
            (suppressed, 'Non-Maximum Suppression'),
            (strong + weak, 'Double Threshold'),
            (final_edges, 'Final Edges')
        ]

        # Display all images in separate windows
        for img, title in images:
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                display_img = np.uint8(np.clip(img, 0, 255))
            else:
                display_img = img
            cv2.imshow(title, display_img)

        print("Press 'q' to close all windows.")
        # Wait until 'q' key is pressed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)

