import cv2
import numpy as np
import stag
import pickle
from datetime import datetime

def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
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
    if corners is None or ids is None:
        return

    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)
        cv2.putText(image, f'ID: {id_}', (centroid_x - (-40), centroid_y - (-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

        # Calcular as dimensões da área de varredura
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20  # Assumindo que o marcador tem 20mm de largura
        crop_width = int(75 * pixel_size_mm)  # Largura da área de varredura em pixels
        crop_height = int(50 * pixel_size_mm)  # Altura da área de varredura em pixels
        crop_y_adjustment = int(10 * pixel_size_mm)  # Ajuste vertical

        # Definir as coordenadas da área de varredura
        x_min = max(centroid_x - crop_height // 2, 0)
        x_max = min(centroid_x + crop_height // 2, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)  # Correção para y_max com base no ajuste y

        # Desenhar a área de varredura
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
        cv2.putText(image, 'Scan Area', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)


def read_contour_signature(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def match_contour(image, contour):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_similarity = 0

    for test_contour in contours:
        similarity = cv2.matchShapes(contour, test_contour, 1, 0.0)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = test_contour

    if best_match is not None:
        cv2.drawContours(image, [best_match], -1, (0, 0, 255), 3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_contour_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"Match encontrado! Similaridade: {max_similarity:.2%}")
        print(f"Imagem salva como {filename}")
        return True, max_similarity
    return False, None

def main():
    image_path = input("Por favor, insira o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(image_path)

    if image is not None:
        display_markers(image, corners, ids)
        cv2.imshow('Marcadores STag', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        selected_id = int(input("Digite o ID do STag para verificação de contorno: "))
        signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
        input_contour = read_contour_signature(signature_path)

        match_found, similarity = match_contour(image, input_contour)
        if match_found:
            cv2.imshow('Match de Contorno Encontrado', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Nenhum match encontrado.")

if __name__ == "__main__":
    main()
