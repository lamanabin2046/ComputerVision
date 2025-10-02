import cv2
import numpy as np
import matplotlib.pyplot as plt

class HoughLineDetector:
    def __init__(self, image_path, canny_threshold1=50, canny_threshold2=150, hough_threshold=150):
        """
        Hough Line Detector using standard Hough Transform (polar coordinates)
        """
        self.image_path = image_path
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.hough_threshold = hough_threshold

        # Load grayscale image
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image from '{image_path}'")
        self.edges = None
        self.lines = None
        self.color_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    # Step 1: Canny edge detection
    def detect_edges(self):
        self.edges = cv2.Canny(self.image, self.canny_threshold1, self.canny_threshold2)

    # Step 2: Standard Hough Transform (polar coordinates)
    def detect_lines(self):
        if self.edges is None:
            self.detect_edges()
        self.lines = cv2.HoughLines(self.edges, rho=1, theta=np.pi/180, threshold=self.hough_threshold)

    # Step 3: Draw infinite-length lines (using ±1000 as in your snippet)
    def draw_lines(self):
        if self.lines is not None:
            for rho, theta in self.lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                # Endpoints far away
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(self.color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display results
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(self.edges, cmap='gray')
        plt.title('Canny Edges')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        plt.title('Hough Line Transform')
        plt.axis('off')
        plt.show()

    # Full pipeline
    def process(self):
        self.detect_edges()
        self.detect_lines()
        self.draw_lines()


# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    image_path = 'images/road_rural.jpg'
    line_detector = HoughLineDetector(image_path)
    line_detector.process()
