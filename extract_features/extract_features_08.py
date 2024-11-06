import cv2
import numpy as np
import matplotlib.pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    
    '''Estratégia de execução class ExtractFeatures: 
        1 - Carrega a imagem e reconhecer a stag passada como parâmetro na chamada da classe; 
          no método "def detect_stag(self)" medir o stag e retornar id, corners e medidas das arestas em milímetros;
        2 - Normalizar a imagem com base nas stags com uso da função "def homogenize_image_based_on_corners(self)"
          executar correção de posicionamento de stag(eixos coordenados) e executando homogeneização da imagem, warped se necessário
          salvar a imagem_homogeneized+nºcrescente na pasta img_homogeneized através da função "def save_features_separated (self)";
        3 - Com base na medida da stag, executar a função "def display_scan_areas_by_markers(self)" para achar a scan_area referen
          te a área de atuação do processamento de imagem;
          salvar a scan_area+nºcrescente na pasta scan_areas através da função "def save_features_separated (self)";
        4 - Criar a máscara do objeto com uso de threshold para isolar o medicamento corretamente usando a função "def create_mask(self, img)
          salvar a mask+nºcrescente na pasta medicine_mask através da função "def save_features_separated (self)";"
        5 - Mapear o contorno a partir da máscara com a função "def find_and_draw_contours(self, homogeneized_img, mask)"
          salvar a contour+nºcrescente na pasta contorno através da função "def save_features_separated (self)";
        6 - Pegar o contorno salvo e transforma-lo em vetores do tipo chain_code, livre informações de localização matriz da imagem
          com uso da função "def create_chain_code(self)"; salvar a chain_code+nºcrescente na pasta contorno_chain_code através da função 
        "def save_features_separated (self)";
        7 - Medir o tamanho do medicamento em suas dimensões (altura x largura) com base na imagem homogeneizada, medidas extraídas
          do stag(20mm), contorno e plotar as medidas no meio da imagem; salvar a measures+nºcrescente na pasta measures através da função
          "def save_features_separated (self)";    
        8 - cropar scan_area com uso da função "def crop_scan_area(self)";
          salvar cropped_area+nºcrescente na pasta cropped através da função "def save_features_separated (self)";"
        9 - remover background da cropped_area com uso da função  " def remove_background(self, image_np_array)";
          extrair o png e salva-lo como medicine_png+nºcrescente na pasta template através da função "def save_features_separated (self)";
        10 - extrai 1 histograma colorido e 1 em escala de cinza com uso da função def histogram_by_template();
          salva na forma hist_color+nºcrescente e hist_gray+nºcrescente na pasta histogram através da função "def save_features_separated (self)";
        11 - extrai a textura do medicamento com base na análise dos histogramas do medicamento em png para criar ambiente com diferentes texturas usando a função def texture_define();
          salva na forma texture_color+nºcrescente e texture_gray+nºcrescente na pasta texture através da função "def save_features_separated (self)";               
          '''
    
    
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.image = None
        self.corners = None
        self.ids = None
        self.homogenized_image = None
        self.scan_areas = {}

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

    def homogenize_image_based_on_corners(self):
        if self.corners is None:
            print("Corners not detected.")
            return False
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
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (max_width, max_height))
        return True

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

        return True
    
    def crop_scan_area(self):
        if self.stag_id not in self.scan_areas:
                print(f'ID {self.stag_id} não encontrado.')
                return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        return self.homogenized_image[y_min:y_max, x_min:x_max]
    
    def remove_background(self, image_np_array):
        is_success, buffer = cv2.imencode(".jpg", image_np_array)
        if not is_success:
            raise ValueError("Falha ao codificar a imagem para remoção de fundo.")
        output_image = remove(buffer.tobytes())
        img = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Falha ao decodificar a imagem processada.")
        return img

    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        return cv2.inRange(img, lower_bound, upper_bound)
    
    def find_and_draw_contours(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 1)
        return img_with_contours, contours
    
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

            # Normalize the direction to the nearest cardinal point
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

        return chain_code
    
    def draw_chain_code(self, img, contour, chain_code):
        # Define the starting point
        start_point = tuple(contour[0][0])
        current_point = start_point
        
        # Moves correspond to chain code directions
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

        # Draw each segment based on the chain code
        for code in chain_code:
            dx, dy = moves[code]
            next_point = (current_point[0] + dx, current_point[1] + dy)
            cv2.line(img, current_point, next_point, (255, 255, 255), 1)  # Draw white line
            current_point = next_point
        
        return img




    def medicine_measures(self, img, contours):
        if not contours:
            print("No contours found.")
            return None

        stag_width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
        px_to_mm_scale = 20 / stag_width_px

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            width_mm = w * px_to_mm_scale
            height_mm = h * px_to_mm_scale

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(img, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return img


if __name__ == "__main__":
    image_path = ".\\frames\\thiago_fotos_10_features_morning\\img_1_010.jpg"
    stag_id = 1
    processor_image = ExtractFeatures(image_path, stag_id)
    if processor_image.detect_stag() and processor_image.homogenize_image_based_on_corners():
        if processor_image.display_scan_area_by_markers():
            cropped_img = processor_image.crop_scan_area()
            obj_rm_bg = processor_image.remove_background(cropped_img)
            mask = processor_image.create_mask(obj_rm_bg)
            desenho, contours = processor_image.find_and_draw_contours(obj_rm_bg, mask)
            medicine_measures = processor_image.medicine_measures(desenho, contours)
            if contours:
                for contour in contours:
                    chain_code = processor_image.compute_chain_code(contour)  # individual contour
                    draw_chain_code = processor_image.draw_chain_code(desenho, contour, chain_code)
                    print("Chain Code:", chain_code)
            plt.figure()
            plt.imshow(cv2.cvtColor(processor_image.homogenized_image, cv2.COLOR_BGR2RGB))
            plt.figure()
            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(obj_rm_bg, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(desenho, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(draw_chain_code, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.figure()
            plt.imshow(cv2.cvtColor(medicine_measures, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("Failed to display markers.")
    else:
        print("Failed to process image.")
