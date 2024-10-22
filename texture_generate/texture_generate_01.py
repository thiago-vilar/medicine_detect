import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from rembg import remove
import stag
import os
import pickle

def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
    '''Detecta as stags presentes na imagem e retorna os cantos e IDs dos marcadores, junto com a imagem lida.'''
    image = cv2.imread(image_path)
    if image is None:
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
    '''Adiciona marcadores visuais para stags detectadas na imagem e define áreas de varredura baseadas na localização dos marcadores.'''
    if corners is None or ids is None:
        return {}

    scan_areas = {}
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        cv2.polylines(image, [corner], True, (0, 255, 0), 1)
        cv2.putText(image, f'ID: {id_}', (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20  
        crop_width = int(75 * pixel_size_mm)
        crop_height = int(50 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        x_min = max(centroid_x - crop_height // 2, 0)
        x_max = min(centroid_x + crop_height // 2, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
        cv2.putText(image, 'Scan Area', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

        scan_areas[id_] = (x_min, x_max, y_min, y_max)

    return scan_areas

def crop_scan_area(image, scan_areas, selected_id):
    '''Corta a imagem na área de varredura selecionada pelo usuário com base no ID.'''
    if selected_id not in scan_areas:
        print(f"ID {selected_id} não encontrado.")
        return None
    x_min, x_max, y_min, y_max = scan_areas[selected_id]
    return image[y_min:y_max, x_min:x_max]

def remove_background(image_np_array):
    '''Remove o fundo de uma imagem numpy array, mantendo o canal alpha se presente.'''
    is_success, buffer = cv2.imencode(".jpg", image_np_array)
    if not is_success:
        raise ValueError("Falha ao codificar a imagem para remoção de fundo.")

    output_image = remove(buffer.tobytes())
    img = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img

def convert_to_grayscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def extract_texture_features(image_np_array):
    '''Extrai características de textura usando Local Binary Patterns.'''
    radius = 3
    n_points = 8 * radius
    method = 'uniform'
    
    lbp_image = local_binary_pattern(image_np_array, n_points, radius, method)
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    
    return lbp_image, hist

def extract_texture_by_gabor(img_gray):
    '''Extrai características de textura usando Gabor.'''
    ksize = (21, 21)  # Tamanho do kernel Gabor
    sigma = 8.0       # Desvio padrão da função gaussiana
    theta = np.pi/4   # Orientação do kernel
    lambd = 10.0      # Comprimento de onda do sinusóide
    gamma = 0.5       # Fator de aspecto espacial
    psi = 0           # Deslocamento de fase
    ktype = cv2.CV_32F  # Tipo de kernel

    g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
    filtered_img = cv2.filter2D(img_gray, cv2.CV_8UC3, g_kernel)
    return filtered_img

def main():
    filepath = input("Digite o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(filepath)
    if corners is not None:
        scan_areas = display_markers(image, corners, ids)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Selecione o ID da área de varredura")
        plt.axis('off')
        plt.show()
        selected_id = int(input("Digite o ID da área de varredura a ser processada: "))
        cropped_image = crop_scan_area(image, scan_areas, selected_id)
        if cropped_image is not None:
            bg_removed_image = remove_background(cropped_image)
            gray_image = convert_to_grayscale(bg_removed_image)
            lbp_image, hist = extract_texture_features(gray_image)
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(lbp_image, cmap='gray')
            plt.title("Imagem LBP")
            plt.axis('off')
            plt.subplot(122)
            plt.bar(np.arange(len(hist)), hist)
            plt.title("Histograma LBP")
            plt.show()

            texture_02 = extract_texture_by_gabor(gray_image)
            plt.imshow(texture_02, cmap='gray')
            plt.title("Textura Gabor")
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    main()
