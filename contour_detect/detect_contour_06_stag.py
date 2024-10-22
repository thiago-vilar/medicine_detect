import cv2
import numpy as np
import os
import stag
import pickle

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

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_scan_area(image, scan_area):
    x_min, x_max, y_min, y_max = scan_area
    return cv2.resize(image[y_min:y_max, x_min:x_max], (100, 100))

def load_contour(contour_path):
    if contour_path.endswith('.pkl'):
        with open(contour_path, 'rb') as file:
            contour = pickle.load(file)
    else:
        raise ValueError("Formato de arquivo de contorno não suportado. Use .pkl.")
    return contour

def match_contour(scan_area_image, contour):
    return cv2.matchShapes(scan_area_image, contour, 1, 0.0)

def check_and_draw_contour(image, scan_area, contour, threshold=0.2):
    similarity_score = match_contour(image[scan_area[2]:scan_area[3], scan_area[0]:scan_area[1]], contour)
    cv2.rectangle(image, (scan_area[0], scan_area[2]), (scan_area[1], scan_area[3]), (0, 255, 0), 2)
    cv2.putText(image, f"Similaridade: {similarity_score*100:.2f}%", (scan_area[0], scan_area[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return similarity_score

def save_image_in_results(image, directory="Resultados", base_filename="resultado"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    i = 0
    while os.path.exists(os.path.join(directory, f"{base_filename}{i}.png")):
        i += 1
    filename = os.path.join(directory, f"{base_filename}{i}.png")
    cv2.imwrite(filename, image)
    return filename

def main():
    image_path = input("Digite o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(image_path)
    scan_areas = display_markers(image, corners, ids)
    cv2.imshow("Image with Markers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ids is not None:
        selected_id = int(input("Digite o ID do marcador para escanear o contorno: "))
        gray_image = convert_to_grayscale(image)
        contour_path = input("Digite o caminho do arquivo de contorno (.pkl): ")
        loaded_contour = load_contour(contour_path)
        scan_area = scan_areas[selected_id]
        scan_area_image = normalize_scan_area(gray_image, scan_area)
        similarity_score = check_and_draw_contour(gray_image, scan_area, loaded_contour)
        save_path = save_image_in_results(gray_image)
        print(f"Imagem salva em: {save_path}")
        cv2.imshow("Resultado final", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
