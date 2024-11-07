import cv2
import numpy as np
import matplotlib.pyplot as plt
import stag

class ExtractFeatures:
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = None
        self.corners = None
        self.ids = None

    def detect_stag(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print("Error loading image.")
            return None
        config = {'libraryHD': 17, 'errorCorrection': -1} # STag library
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None:
            index = np.where(self.ids == self.stag_id)[0]
            if len(index) > 0:
                index = index[0]
                self.corners = self.corners[index].reshape(-1, 2)
                return self.corners
            else:
                print("Marker with ID", self.stag_id, "not found.")
        else:
            print("No markers found.")
        return None

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
        max_width = self.image.shape[1]
        max_height = self.image.shape[0]
        homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (max_width, max_height))
        return homogenized_image
    
    


    

if __name__ == "__main__":
    image_path = "./frames/img_0_010.jpg"
    stag_id = 0
    processor_image = ExtractFeatures(image_path, stag_id)
    corners = processor_image.detect_stag()
    if corners is not None:
        print("Original Corners of STag ID", stag_id, ":\n", corners)

        homogenized_image = processor_image.homogenize_image_based_on_corners()
        if homogenized_image is not None:
            plt.imshow(cv2.cvtColor(homogenized_image, cv2.COLOR_BGR2RGB))
            plt.title("Homogenized Image")
            plt.show()
    else:
        print("Failed to detect corners.")


