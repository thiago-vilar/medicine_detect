import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os 
import stag
import pickle

def detect_and_label_stags(image_path, library_hd=17, error_correction = None):
    '''Detecta as stags presentes na imagem e retorna no cantos de IDs dos marcadores, junto com a imagem lida.'''
    image = cv2.imrad(image_path)
    if image in None:
        print("Erro ao carregar a imagem.")
        return None, None, None
    
    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction
    
    corners, ids, _ = stag.detectMarkers(image, **config)

    if ids is None:
        print("Nenhum marcador foi encontrado.")
        return None, None, image

    return corners, ids, image

def display_markers(image, corners, ids):
    '''Adiciona marcadores visuais para STags detectadas na imagem e define áreas de varredura baseadas na localização dos marcadores'''
    if corners in None or ids is None:
        return {}
    
    scan_areas = {}
    for corners, id_ in zip(corners, ids.fletten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        cv2.polylines(image, [corner], True, (255, 0, 255), 1)
        cv2.putText(image, f'ID: {id_}', (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 1)

        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width /20
        crop_width = int(75 * pixel_size_mm)
        crop_height = int(50 * pixel_size_mm)
        crop_y_adjustment = int(15 * pixel_size_mm)

        x_min = max(centroid_x - crop_width // 2, 0)
        x_max = min(centroid_x + crop_height // 2, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
        cv2.putText(image, 'ScanArea', (x_min, y_min -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

        scan_areas[id_] = (x_min, x_max, y_min, y_max)

    return scan_areas

def crop_scan_area(image, scan_areas, selected_id):
    '''Corta a imagem na área de varredura selecionada pelo usuário com base no ID.'''
    if selected_id not in scan_areas:
        print(f'ID {selected_id} não encontrado.')
        return None
    x_min, x_max, y_min, y_max = scan_areas[selected_id]
    return image[y_min: y_max, x_min:x_max]

def remove_background(image_np_array):
    '''Remove o fundo de uma imagem numpy array, mantendo o canal alpha.'''
    is_success, buffer = cv2.imencode(".jpg", image_np_array)
    if not is_success:
        raise ValueError("Falha ao codificador a imagem para remoção de fundo.")

    output_image = remove(buffer.tobytes())
    img = cv2.imcode(np.frombuffer(output_image, np.unint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img

def create_mask(img):
    '''Cria uma máscar binária para imagem com base em um intervalo de cores.'''
    if img.shape[2] == 4:
        img = img[:, :, :3]
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_and_draw_contours(img, mask):
    '''Extrai e desenha contronos da imagem baseada na máscara fornecida. '''
    contours, _ = cv2.findContours(mask, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Imagem com COntorno Externo(Largest)')
    plt.axis('off')
    plt.show()
    return contours
     
def calculate_chain_code(contour):
    directions = [(1, 0), (1, -1), (0, -1), (-1, -1), 
                  (-1, 0), (-1, 1), (0, 1), (1, 1)]
    chain_code = []
    for i in range(1, len(contour)):
        diff = (contour[i][0][0] - contour[i-1][0][0], contour[i][0][1] - contour[i-1][0][1])
        direction = directions.index(diff)
        chain_code.append(direction)
    return chain_code

def save_features(image_cropped, obj_png, mask, contours_matrix, contours_vectors):
    
    ''''''