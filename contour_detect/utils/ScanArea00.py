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
        return {}

    scan_areas = {}
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        # Desenha o contorno e o ID do marcador
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)
        cv2.putText(image, f'ID: {id_}', (centroid_x, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calcular e definir as dimensões da área de varredura
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20  # Assumindo que o marcador tem 20mm de largura
        crop_width = int(75 * pixel_size_mm)
        crop_height = int(50 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        # Coordenadas da área de varredura
        x_min = max(centroid_x - crop_height // 2, 0)
        x_max = min(centroid_x + crop_height // 2, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)

        # Definir e desenhar a área de varredura
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
        cv2.putText(image, 'Scan Area', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Armazenar área de varredura no dicionário
        scan_areas[id_] = (x_min, x_max, y_min, y_max)

    return scan_areas

def read_contour_signature(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def match_contour(image, contour, scan_area):
    x_min, x_max, y_min, y_max = scan_area
    crop_img = image[y_min:y_max, x_min:x_max]
    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar todos os contornos encontrados na área de varredura para visualização
    for cnt in contours:
        # Ajuste para coordenadas globais antes de desenhar
        adjusted_contour = cnt + np.array([[x_min, y_min]])
        cv2.drawContours(image, [adjusted_contour], -1, (255, 255, 0), 2)  # Amarelo para destaque

    best_match = None
    min_similarity = float('inf')
    for test_contour in contours:
        similarity = cv2.matchShapes(contour, test_contour, 1, 0.0)
        if similarity < min_similarity:
            min_similarity = similarity
            best_match = test_contour

    if best_match is not None:
        adjusted_contour = best_match + np.array([[x_min, y_min]])
        cv2.drawContours(image, [adjusted_contour], -1, (0, 255, 0), 3)  # Verde para o melhor match
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_contour_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"Match encontrado! Similaridade: {min_similarity:.2%}")
        print(f"Imagem salva como {filename}")
        return True, min_similarity, adjusted_contour
    return False, None, None



def main():
    image_path = input("Por favor, insira o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(image_path)

    if image is not None:
        scan_areas = display_markers(image, corners, ids)
        cv2.imshow('Marcadores STag', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        selected_id = int(input("Digite o ID do STag para verificação de contorno: "))
        signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
        input_contour = read_contour_signature(signature_path)

        if selected_id in scan_areas:
            match_found, similarity, contours = match_contour(image, input_contour, scan_areas[selected_id])
            if match_found:
                cv2.imshow('Match de Contorno Encontrado', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Nenhum match encontrado.")
        else:
            print("ID não encontrado na imagem.")


if __name__ == "__main__":
    main()

