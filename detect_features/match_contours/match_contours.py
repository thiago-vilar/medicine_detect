import os
import cv2
import numpy as np
import stag
from rembg import remove
import pickle

class ExtractFeatures:
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

    def detect_stag(self):
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    def homogenize_image_based_on_corners(self):
        if self.corners is None:
            return None
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        aligned_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(self.corners, aligned_corners)
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (self.image.shape[1], self.image.shape[0]))
        return self.homogenized_image

    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        return cv2.inRange(img, lower_bound, upper_bound)

    def find_and_draw_contours(self, mask):
        """Finds and draws only the largest contour around the foreground object based on the mask and saves the image with alpha transparency."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.size > 0:
                mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                mask_with_contours[:, :, 3] = mask  
                cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 0, 255, 255), 2)  
                #Save             
                directory = 'features/contours'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_number = 0
                while os.path.exists(f'{directory}/contour_{file_number}.png'):
                    file_number += 1
                cv2.imwrite(f'{directory}/contour_{file_number}.png', mask_with_contours)
                print(f'Contour image saved as contour_{file_number}.png in {directory}')
                return mask_with_contours, largest_contour
        else:
            return None

    def compute_chain_code(self, contour):
        ''' Calculates chain code for object contours for shape analysis. '''
        start_point = contour[0][0]
        current_point = start_point
        chain_code = []
        moves = {
            (-1, 0) : 3,
            (-1, 1) : 2,
            (0, 1)  : 1,
            (1, 1)  : 0,
            (1, 0)  : 7,
            (1, -1) : 6,
            (0, -1) : 5,
            (-1, -1): 4
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
        # Save
        directory = 'features/chain_code'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_number = 0
        file_path = os.path.join(directory, f'chain_code_{file_number}.pkl')
        while os.path.exists(file_path):
            file_number += 1
            file_path = os.path.join(directory, f'chain_code_{file_number}.pkl')
        
        with open(file_path, 'wb') as file:
            pickle.dump(chain_code, file)
        print(f"Chain code saved to {file_path}")
        print("Chain code sequence:", chain_code)

        return chain_code, len(chain_code)

    def draw_chain_code(self, img_med, contour, chain_code):
        ''' Draws the chain code on the image to visually represent contour direction changes. '''
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
            cv2.line(img_med, current_point, next_point, (255, 255, 255), 1)
            current_point = next_point
        return img_med, len(chain_code)
class LoadContours:
    def __init__(self, contour_path, chain_code_path):
        self.contour_path = contour_path
        self.chain_code_path = chain_code_path

    def load_data(self):
        with open(self.contour_path, 'rb') as f:
            self.contours = pickle.load(f)
        with open(self.chain_code_path, 'rb') as f:
            self.chain_codes = pickle.load(f)
            

def calculate_contour_iou(contour1, contour2):
    contour2_resized = cv2.resize(contour2, (contour1.shape[1], contour1.shape[0]))
    intersection = np.logical_and(contour1, contour2_resized)
    union = np.logical_or(contour1, contour2_resized)
    return np.sum(intersection) / np.sum(union)

def calculate_chain_code_iou(chain_code1, chain_code2):
    chain_code2_resized = cv2.resize(chain_code2, (chain_code1.shape[1], chain_code1.shape[0]))
    intersection = np.logical_and(chain_code1, chain_code2_resized)
    union = np.logical_or(chain_code1, chain_code2_resized)
    return np.sum(intersection) / np.sum(union)


if __name__ == "__main__":
    #