import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
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

    def crop_scan_area(self):
        """Crops the defined scan area from the homogenized image and saves it locally."""
        if self.stag_id not in self.scan_areas:
            print(f'ID {self.stag_id} not found.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y_min:y_max, x_min:x_max]
        # #Save
        # if not os.path.exists('features/cropped_imgs'):
        #     os.makedirs('features/cropped_imgs')
        # file_number = 0
        # while os.path.exists(f'features/cropped/img_cropped_{file_number}.jpg'):
        #     file_number += 1
        # cv2.imwrite(f'features/cropped/img_cropped_{file_number}.jpg', cropped_image)
        # print(f'Image saved as img_cropped_{file_number}.jpg')
        return cropped_image
    
    def sobel_filter(self, img_gaussian_gray):
        grad_x = cv2.Sobel(img_gaussian_gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gaussian_gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        img_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return img_sobel
    
    def otsu(self, img_med):
        # Assegurando que a imagem está em escala de cinza
        if len(img_med.shape) == 3 and img_med.shape[2] == 3:  # Checa 3 canais
            img_gray = cv2.cvtColor(img_med, cv2.COLOR_BGR2GRAY)
        elif len(img_med.shape) == 2 or (len(img_med.shape) == 3 and img_med.shape[2] == 1):
            img_gray = img_med  
        else:
            raise ValueError("Formato de imagem não suportado para o processamento de Otsu.")

        # Aplicando o threshold de Otsu
        _, otsu_binarized_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_binarized_image
    
    def bina_img (self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = 140
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def erode(self, bina_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        num_erosion_iterations = 1.5
        erode = None  # Definir erode como None inicialmente para evitar referência antes da atribuição
        if bina_img is not None and num_erosion_iterations is not None:  # Verificação mais clara
            erode = cv2.erode(bina_img, kernel, iterations=num_erosion_iterations)
        return erode


    def erode_outer_contour(self, image):
        # Encontrar contornos na imagem, considerando apenas o contorno externo
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Criar uma máscara branca do mesmo tamanho da imagem original
        mask = np.zeros_like(image)
        
        # Desenhar o contorno mais externo na máscara
        if contours:
            # Assumindo que o contorno mais externo é o maior
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, (255), -1)  

        # Definir o kernel para erosão e o número de iterações
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        num_erosion_iterations = 1
        
        # Aplicar erosão apenas na máscara do contorno
        eroded_mask = cv2.erode(mask, kernel, iterations=num_erosion_iterations)
        
        # Aplicar a máscara erodida de volta à imagem original
        # Isso substituirá apenas as regiões do contorno original
        eroded_image = np.where(eroded_mask == 255, 255, image)

        return eroded_image


    def dilate(self, bina_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        num_dilation_iterations = 1.5
        dilated = None
        if bina_img is not None and  num_dilation_iterations is not None:
            dilated = cv2.dilate(bina_img, kernel, iterations=num_dilation_iterations)
        return dilated

    def remove_background(self, image_np_array):
        """Removes the background from the cropped scan area and saves the image with alpha channel."""
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        # # Save
        # if not os.path.exists('features/medicine_png'):
        #     os.makedirs('features/medicine_png')
        # file_number = 0
        # while os.path.exists(f'features/medicine_png/medicine_{file_number}.png'):
        #     file_number += 1
        # cv2.imwrite(f'features/medicine_png/medicine_{file_number}.png', img_med)
        # print(f'Image saved as medicine_{file_number}.png with transparent background')
        return img_med

    def calculate_color_histograms(self, image):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title('Histograma de Cores')
        plt.show()

    def segment_colors(self, image, k=7):
        reshaped_image = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        # save_dir = 'segmented_color_img'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # file_number = 0
        # file_path = os.path.join(save_dir, f'segmented_{file_number}.png')
        # while os.path.exists(file_path):
        #     file_number += 1
        #     file_path = os.path.join(save_dir, f'segmented_{file_number}.png')
        
        # cv2.imwrite(file_path, segmented_image)
        # print(f'Image saved as {file_path}')

        return segmented_image

    # restante das definições de métodos...

if __name__ == "__main__":
    # image_path = ".\\frames\\new_test_light\\thiago_fotos_10_down_lighton_ampoules\\color_c1.jpg"
    image_path = ".\\frames\\canaleta_azul\\img_10_004.jpg"
    stag_id = 10
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

                   # Segmentando cores e mostrando resultado
                    segmented = processor.segment_colors(cropped)
                    plt.imshow(segmented, cmap='nipy_spectral')  
                    plt.colorbar()  
                    plt.title('Imagem Segmentada N-cores')
                    plt.show()

                    remove_bg = processor.remove_background(segmented)
                    plt.imshow(cv2.cvtColor(remove_bg, cv2.COLOR_BGR2RGB))
                    plt.title('Remove bg de IMG Segmentada')
                    plt.show()

                    remove_bg_by_cropped = processor.remove_background(cropped)
                    plt.imshow(cv2.cvtColor(remove_bg_by_cropped, cv2.COLOR_BGR2RGB))
                    plt.title('Remove bg img cropped original')
                    plt.show()

                    segmented_2 = processor.segment_colors(remove_bg_by_cropped)
                    plt.imshow(segmented_2, cmap='nipy_spectral')
                    plt.colorbar()  
                    plt.title('Kmeans - obj segmentado') 
                    plt.show()

                    otsu_img = processor.bina_img(segmented_2)
                    otsu_img=~otsu_img
                    plt.imshow(otsu_img)
                    plt.title('bina') 
                    plt.show()

                    last_remove = processor.remove_background(otsu_img )
                    plt.imshow(last_remove)
                    plt.title('Last Remove TRY') 
                    plt.show()

            
    else:
        print("Stag detection failed.")
