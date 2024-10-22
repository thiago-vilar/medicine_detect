import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
import stag
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

def create_mask(img):
    '''Cria uma máscara binária para a imagem com base em um intervalo de cores.'''
    if img.shape[2] == 4:  # Se houver 4 canais, remove o canal alpha
        img = img[:, :, :3]
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_and_draw_contours(img, mask):
    '''Extrai e desenha contornos da imagem baseada na máscara fornecida.'''
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("Imagem com Contorno Externo(Largest)")
    plt.axis('off')
    plt.show()
    return contours

def save_image_and_contours(img, contours):
    '''
    Salva a imagem processada e o maior contorno extraído em arquivos separados.
    A função garante que a pasta de destino seja criada se não existir, salva a imagem processada,
    e identifica e salva o maior contorno encontrado, assegurando que esteja no formato correto (N, 2).
    '''
    directory = "processed_features"
    os.makedirs(directory, exist_ok=True)
    num_files = len(os.listdir(directory)) + 1  

    image_filename = os.path.join(directory, f"image{num_files}.png")
    cv2.imwrite(image_filename, img)
    print(f"Imagem salva: {image_filename}")

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour = largest_contour.reshape(-1, 2)  
        contour_filename = os.path.join(directory, f"contour{num_files}.pkl")
        with open(contour_filename, 'wb') as f:
            pickle.dump(largest_contour, f)
        print(f"Contorno salvo: {contour_filename}")

    return image_filename, contour_filename


def main():
    '''Executa o fluxo completo de processamento de imagem desde a detecção de marcadores até o salvamento de imagens e contornos processados.'''
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
            mask = create_mask(bg_removed_image)
            plt.imshow(mask, cmap='gray')
            plt.title("Máscara Criada Após a Remoção de Fundo")
            plt.axis('off')
            plt.show()
            contours = extract_and_draw_contours(bg_removed_image, mask)
            save_image_and_contours(bg_removed_image, contours)
            plt.imshow(cv2.cvtColor(bg_removed_image, cv2.COLOR_BGR2RGB))
            plt.title("Imagem Processada")
            plt.axis('off')
            plt.show()
    else:
        print("Nenhum marcador foi detectado na imagem.")

if __name__ == "__main__":
    main()
