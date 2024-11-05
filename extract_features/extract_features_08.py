import cv2
import numpy as np
import matplotlib.pyplot as plt
import stag
from rembg import remove

'''Estratégia de execução linear da classe: 
        1 - Carrega a imagem e reconhecer a stag passada como parâmetro na chamada da classe; 
        no método "def detect_stag(self)" medir o stag e retornar id, corners e medidas das arestas em milímetros;
        2 - Normalizar a imagem com base nas stags com uso da função "def homogenize_image_based_on_corners(self)"
        executar correção de posicionamento de stag(eixos coordenados) e executando homogeneização da imagem, warped se necessário
        salvar a imagem_homogeneized+nºcrescente na pasta img_homogeneized através da função "def save_features_separated (self)";
        3 - Com base na medida da stag, executar a função "def display_scan_areas_by_markers(self)" para achar a scan_area referente a área de atuação do processamento de imagem;
        salvar a scan_area+nºcrescente na pasta scan_areas através da função "def save_features_separated (self)";
        4 - Criar a máscara do objeto com uso de threshold para isolar o medicamento corretamente usando a função "def create_mask(self, img)
        salvar a mask+nºcrescente na pasta medicine_mask através da função "def save_features_separated (self)";"
        5 - Mapear o contorno a partir da máscara com a função "def find_and_draw_contours(self, homogeneized_img, mask)"
        salvar a contour+nºcrescente na pasta contorno através da função "def save_features_separated (self)";
        6 - Pegar o contorno salvo e transforma-lo em vetores do tipo chain_code, livre informações de localização matriz da imagem
        com uso da função "def create_chain_code(self)"; salvar a chain_code+nºcrescente na pasta contorno_chain_code através da função 
        "def save_features_separated (self)";
        7 - Medir o tamanho do medicamento em suas dimensões (altura x largura) com base na imagem homogeneizada, medidas extraídas do stag(20mm), contorno
        e plotar as medidas no meio da imagem; salvar a measures+nºcrescente na pasta measures através da função "def save_features_separated (self)";    
        8 - cropar scan_area com uso da função "def crop_scan_area(self)";
        salvar cropped_area+nºcrescente na pasta cropped através da função "def save_features_separated (self)";"
        9 - remover background da cropped_area com uso da função  " def remove_background(self, image_np_array)";
        extrair o png e salva-lo como medicine_png+nºcrescente na pasta template através da função "def save_features_separated (self)";
        10 - extrai 1 histograma colorido e 1 em escala de cinza com uso da função def histogram_by_template();
        salva na forma hist_color+nºcrescente e hist_gray+nºcrescente na pasta histogram através da função "def save_features_separated (self)";
        11 - extrai a textura do medicamento com base na análise dos histogramas do medicamento em png para criar ambiente com diferentes texturas usando a função def texture_define();
        salva na forma texture_color+nºcrescente e texture_gray+nºcrescente na pasta texture através da função "def save_features_separated (self)";               
        '''

class ExtractFeatures:
    def __init__(self, image_path, stag_id):
        self.image_path = image_path
        self.stag_id = stag_id
        self.stag_measure = 20 #'mm'
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
    
    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        lower_bound = np.array([30, 30, 30])
        upper_bound = np.array([255, 255, 255])
        return cv2.inRange(img, lower_bound, upper_bound)
    
    def find_and_draw_contours(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 255), 1)
        return img_with_contours, contours
    
    #def create_chain_code(self):

    def measure_medicine_by_marker(self, img, stag_measure, contours):
        stag_measure = self.stag_measure

        if self.corners is None:
            print("No corners detected for STag.")
            return img
        width_in_pixels = np.linalg.norm(self.corners[0] - self.corners[1])
        pixel_size_cm = 2 / width_in_pixels  # Cada lado do STag tem 2 cm
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            width_cm = w * pixel_size_cm
            height_cm = h * pixel_size_cm
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"{width_cm:.1f} cm x {height_cm:.1f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return img
    
    def display_scan_areas_by_markers(self):
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
    # def histogram_by_template():

    # def texture_define():

    def save_features_separated (self):
        # template png
        # histograma (extraído do png)
        # máscara (extraído do png)
        # cores predominantes (clusters de texturas extraídos do png)
        # contorno vetorizado(chain code)
        # dimensionamento do medicamento (extraído do stag e do template png)
        # morfologia (shape descritors)
        return
   

if __name__ == "__main__":
    image_path = "./frames/img_0_010.jpg"
    stag_id = 0
    processor_image = ExtractFeatures(image_path, stag_id)
    stag_detect =  processor_image.detect_stag()
    homogenized_img = processor_image.homogenize_image_based_on_corners(stag_detect)
    medicine_measure = processor_image.measure_medicine_by_marker( homogenized_img )
    processor_image.display_scan_areas_by_markers()
    cropped_img =  processor_image.crop_scan_area()
    obj_rm_bg = processor_image.remove_background(cropped_img)
    mask = processor_image.create_mask(obj_rm_bg)
    desenho, contour = processor_image.find_and_draw_contours(cropped_img, mask)

    plt.figure()
    plt.imshow(cv2.cvtColor(processor_image.homogenized_image, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.show()
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
    