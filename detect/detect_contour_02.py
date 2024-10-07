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
    scan_areas = {}
    if corners is None or ids is None:
        return scan_areas

    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)
        cv2.putText(image, f'ID: {id_}', (centroid_x - (-40), centroid_y - (-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

        # Definição da área de varredura
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20
        crop_width = int(75 * pixel_size_mm)
        crop_height = int(50 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        x_min = max(centroid_x - crop_height // 2, 0)
        x_max = min(centroid_x + crop_height // 2, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)

        scan_area = ((x_min, y_min), (x_max, y_max))
        scan_areas[id_] = scan_area
        cv2.rectangle(image, scan_area[0], scan_area[1], (255, 0, 255), 1)
        cv2.putText(image, 'Scan Area', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return scan_areas

def read_contour_signature(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def match_contour_in_area(image, scan_area, contour_signature):
    x_min, y_min = scan_area[0]
    x_max, y_max = scan_area[1]
    region_of_interest = image[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_similarity = 0

    for test_contour in contours:
        similarity = cv2.matchShapes(contour_signature, test_contour, 1, 0.0)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = test_contour

    if best_match is not None:
        # Desenhe os contornos diretamente na região de interesse que foi extraída da imagem original
        cv2.drawContours(region_of_interest, [best_match], -1, (0, 0, 255), 3)

        # Atualizar a imagem original com a região de interesse modificada
        image[y_min:y_max, x_min:x_max] = region_of_interest

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_contour_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"Match encontrado! Similaridade: {max_similarity:.2%}")
        print(f"Imagem salva como {filename}")
        cv2.imshow('Match de Contorno Encontrado', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True, max_similarity
    return False, None


def main():
    image_path = input("Por favor, insira o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(image_path)
    if image is not None:
        scan_areas = display_markers(image, corners, ids)
        cv2.imshow('Marcadores STag', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        selected_id = int(input("Digite o ID do STag para a área de varredura: "))
        if selected_id in scan_areas:
            signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
            input_contour = read_contour_signature(signature_path)

            match_found, similarity = match_contour_in_area(image, scan_areas[selected_id], input_contour)
            if not match_found:
                print("Nenhum match encontrado.")
        else:
            print("ID de STag inválido.")

if __name__ == "__main__":
    main()
