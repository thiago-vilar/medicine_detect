import cv2
import numpy as np
import stag
import pickle
from datetime import datetime

def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erro ao carregar a imagem.")
        return None, None, None

    # Aplicando Sobel para extrair contornos
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    image = cv2.magnitude(sobelx, sobely)

    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    corners, ids, _ = stag.detectMarkers(image.astype(np.uint8), **config)
    if ids is None:
        print("Nenhum marcador foi encontrado.")
        return None, None, image

    return corners, ids, image

def display_markers(image, corners, ids):
    scan_areas = {}
    if corners is None or ids is None:
        return scan_areas

    hot_pink = (180, 105, 255)
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))
        cv2.polylines(image, [corner], True, hot_pink, 2)
        cv2.putText(image, f'ID: {id_}', (centroid_x - 40, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, hot_pink, 2)

        # Calcula e desenha áreas de varredura
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        height = np.max(corner[:, 1]) - np.min(corner[:, 1])
        scan_area = ((centroid_x - width, centroid_y - height), (centroid_x + width, centroid_y + height))
        scan_areas[id_] = scan_area
        cv2.rectangle(image, scan_area[0], scan_area[1], hot_pink, 2)
        cv2.putText(image, 'Scan Area', (scan_area[0][0], scan_area[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hot_pink, 1)

    return scan_areas

def read_contour_signature(filename):
    with open(filename, 'rb') as f:
        contour_signature = pickle.load(f)
    return contour_signature

def match_contour_in_area(image, scan_area, contour_signature):
    x_min, y_min, x_max, y_max = *scan_area[0], *scan_area[1]
    region_of_interest = image[y_min:y_max, x_min:x_max]
    contours, _ = cv2.findContours(region_of_interest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_similarity = 0
    red_color = (0, 0, 255)
    for test_contour in contours:
        similarity = cv2.matchShapes(contour_signature, test_contour, 1, 0.0)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = test_contour

    if best_match is not None:
        cv2.drawContours(image, [best_match], -1, red_color, 3, offset=(x_min, y_min))
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
            cv2.imshow('Área de Varredura Selecionada', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  
            signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
            input_contour = read_contour_signature(signature_path)
            match_found, similarity = match_contour_in_area(image, scan_areas[selected_id], input_contour)
            if not match_found:
                print("Nenhum match encontrado.")
        else:
            print("ID de STag inválido.")

if __name__ == "__main__":
    main()


def main():
    image_path = input("Por favor, insira o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(image_path)
    if image is not None:
        scan_areas = display_markers(image, corners, ids)
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
