import cv2
import numpy as np
from matplotlib import pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    ''' Initializes with the path to an image and a specific marker (stag) ID. '''
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image could not be loaded.")
        self.corners = None
        self.ids = None
        self.homogenized_image = None
        self.scan_areas = {}
        self.pixel_size_mm = None  # Millimeters per pixel

    ''' Detects a predefined stag marker in the image using the stag library. '''
    def detect_stag(self):
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            self.calculate_pixel_size_mm()
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    ''' Calculates the pixel size in millimeters based on the detected stag marker. '''
    def calculate_pixel_size_mm(self):
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px  # Assuming the stag is 20 mm wide

    ''' Normalizes the image perspective based on detected stag corners. '''
    def homogenize_image_based_on_corners(self):
        if self.corners is None:
            print("Corners not detected.")
            return None
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        aligned_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(self.corners, aligned_corners)
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (self.image.shape[1], self.image.shape[0]))
        return self.homogenized_image

    ''' Displays the scan area on the homogenized image based on the stag location. '''
    def display_scan_area_by_markers(self):
        if self.homogenized_image is None:
            print("Homogenized image is not available.")
            return None
        
        corner = self.corners.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        cv2.putText(self.homogenized_image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20
        crop_width = int(25 * pixel_size_mm)
        crop_height = int(75 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        x_min = max(centroid_x - crop_width, 0)
        x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
        y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)

        cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
        return self.homogenized_image

    ''' Crops the defined scan area from the homogenized image. '''
    def crop_scan_area(self):
        if self.stag_id not in self.scan_areas:
            print(f'ID {self.stag_id} not found.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y_min:y_max, x_min:x_max]
        return cropped_image

    ''' Removes the background from the cropped scan area using the rembg library. '''
    def remove_background(self, image_np_array):
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        return img_med

    def histogram_view(self, img_med):
        # Split image into RGB channels
        chans = cv2.split(img_med)
        colors = ("r", "g", "b")
        plt.figure()
        for chan, color in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalize histogram
            plt.plot(hist, color=color)
            plt.title(f"{color.capitalize()} Histogram (Normalized)")
            plt.xlabel("Bins")
            plt.ylabel("% of Pixels")
            plt.xlim([0, 256])
        plt.show()

    # Add other methods as previously defined...

if __name__ == "__main__":
    image_path = ".\\frames\\IMG-20241107-WA0031.jpg"
    stag_id = 3
    processor = ExtractFeatures(image_path, stag_id)
    if processor.detect_stag():
        homogenized = processor.homogenize_image_based_on_corners()
        if homogenized is not None:
            plt.imshow(cv2.cvtColor(homogenized, cv2.COLOR_BGR2RGB))
            plt.title('Homogenized Image')
            plt.show()

            marked_image = processor.display_scan_area_by_markers()
            if marked_image is not None:
                plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                plt.title('Marked Scan Area')
                plt.show()

                cropped = processor.crop_scan_area()
                if cropped is not None:
                    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    plt.title('Cropped Scan Area')
                    plt.show()

                    background_removed = processor.remove_background(cropped)
                    if background_removed is not None:
                        plt.imshow(cv2.cvtColor(background_removed, cv2.COLOR_BGR2RGB))
                        plt.title('Background Removed')
                        histogram_go = processor.histogram_view(background_removed)
                        if histogram_go is not None:
                            plt.imshow(cv2.cvtColor(histogram_go, cv2.COLOR_BGR2RGB))
                            plt.title('Histogram Go')
                            plt.show()

    else:
        print("Stag detection failed.")

