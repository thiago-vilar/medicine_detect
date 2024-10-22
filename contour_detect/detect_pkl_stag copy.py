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
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))
        cv2.putText(image, f'ID: {id_}', (centroid_x, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def read_contour_signature(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def match_contour(image, contour, reference_corner):
    # Define ROI based on reference_corner
    x_min, y_min = np.min(reference_corner[:, 0]), np.min(reference_corner[:, 1])
    x_max, y_max = np.max(reference_corner[:, 0]), np.max(reference_corner[:, 1])
    roi = image[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Finding contours in ROI
    contours, _ = cv2.findContours(gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_match = None
    max_similarity = 0

    for test_contour in contours:
        similarity = cv2.matchShapes(contour, test_contour, 1, 0.0)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = test_contour

    if best_match is not None:
        # Convert ROI coordinates to image coordinates
        best_match[:, :, 0] += x_min
        best_match[:, :, 1] += y_min
        cv2.drawContours(image, [best_match], -1, (0, 0, 255), 3)
        cx, cy = np.mean(best_match[:, :, 0]), np.mean(best_match[:, :, 1])
        cv2.putText(image, f"Similaridade: {max_similarity:.2%}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
        selected_index = np.where(ids.flatten() == selected_id)[0][0]
        reference_corner = corners[selected_index]

        signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
        input_contour = read_contour_signature(signature_path)

        match_found, similarity = match_contour(image, input_contour, reference_corner)
        if match_found:
            cv2.imshow('Match de Contorno Encontrado', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Nenhum match encontrado.")

if __name__ == "__main__":
    main()
