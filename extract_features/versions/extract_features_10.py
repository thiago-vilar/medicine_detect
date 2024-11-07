import cv2
import numpy as np
from matplotlib import pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    '''Initializes with the path to an image and a specific marker (stag) ID.'''
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

    '''Detects a predefined stag marker in the image using the stag library.'''
    def detect_stag(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Error loading image.")
            return False
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    '''Normalizes the image perspective based on detected stag corners.'''
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

    '''Displays the scan area on the homogenized image based on the stag location.'''
    def display_scan_area_by_markers(self):
        if self.homogenized_image is None:
            print("Homogenized image is not available.")
            return False
        
        if self.corners is None:
            print("No corners available for processing.")
            return False
        
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
        cv2.putText(self.homogenized_image, 'Scan Area', (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
        return self.homogenized_image

    '''Crops the defined scan area from the homogenized image.'''
    def crop_scan_area(self):
        if self.stag_id not in self.scan_areas:
            print(f'ID {self.stag_id} not found.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y_min:y_max, x_min:x_max]
        return cropped_image

    '''Removes the background from the cropped scan area using the rembg library.'''
    def remove_background(self, image_np_array):
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        return img_med

    '''Creates a binary mask for the foreground object in the image.'''
    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        return mask

    '''Finds and draws contours around the foreground object based on the mask.'''
    def find_and_draw_contours(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 1)
        return img_with_contours, contours

    '''Calculates chain code for object contours for shape analysis.'''
    def compute_chain_code(self, contour):
        start_point = contour[0][0]
        current_point = start_point
        chain_code = []
        moves = {
            (-1, 0) : 3,  # Move to left
            (-1, 1) : 2,  # Move to top-right
            (0, 1)  : 1,  # Move to right
            (1, 1)  : 0,  # Move to bottom-right
            (1, 0)  : 7,  # Move to bottom
            (1, -1) : 6,  # Move to bottom-left
            (0, -1) : 5,  # Move to left
            (-1, -1): 4   # Move to top-left
        }
        for i in range(1, len(contour)):
            next_point = contour[i][0]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
            move = (dx, dy)
            if move in moves:
                chain_code.append(moves[move])
            current_point = next_point
        # Close the loop
        dx = start_point[0] - current_point[0]
        dy = start_point[1] - current_point[1]
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        move = (dx, dy)
        if move in moves:
            chain_code.append(moves[move])
        return chain_code, len(chain_code)

    '''Draws the chain code on the image to visually represent contour direction changes.'''
    def draw_chain_code(self, img, contour, chain_code):
        start_point = tuple(contour[0][0])
        current_point = start_point
        moves = {
            0: (1, 1),    # bottom-right
            1: (0, 1),    # right
            2: (-1, 1),   # top-right
            3: (-1, 0),   # left
            4: (-1, -1),  # top-left
            5: (0, -1),   # left
            6: (1, -1),   # bottom-left
            7: (1, 0)     # bottom
        }
        for code in chain_code:
            dx, dy = moves[code]
            next_point = (current_point[0] + dx, current_point[1] + dy)
            cv2.line(img, current_point, next_point, (255, 255, 255), 1) 
            current_point = next_point
        return img, len(chain_code)

    '''Measures the dimensions of the detected contours and appends to a list.'''
    def medicine_measures(self, img, contours):
        if not contours:
            print("No contours found.")
            return None
        measurements = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            measurements.append((w, h))
        return measurements

if __name__ == "__main__":
    image_path = ".\\frames\\img_0_010.jpg"
    stag_id = 0
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
                        plt.show()

                        mask = processor.create_mask(background_removed)
                        plt.imshow(mask, cmap='gray')
                        plt.title('Mask Created')
                        plt.show()

                        contoured_image, contours = processor.find_and_draw_contours(background_removed, mask)
                        plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
                        plt.title('Contoured Image')
                        plt.show()

                        if contours:
                            for contour in contours:
                                chain_code, chain_length = processor.compute_chain_code(contour)
                                chain_drawn_image, _ = processor.draw_chain_code(contoured_image.copy(), contour, chain_code)
                                plt.imshow(cv2.cvtColor(chain_drawn_image, cv2.COLOR_BGR2RGB))
                                plt.title('Chain Code Drawn')
                                plt.show()
                                print(f"Chain Code: {chain_code}")

                            measurements = processor.medicine_measures(contoured_image, contours)
                            plt.imshow(cv2.cvtColor(measurements, cv2.COLOR_BGR2RGB))
                            plt.title('measures')
                            plt.show()
                            print("Medicine measurements:", measurements)
    else:
        print("Stag detection failed.")