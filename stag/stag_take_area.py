import cv2
import numpy as np
import stag 

def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Configurar a detecção de marcadores STag
    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    # Detectar marcadores na imagem
    corners, ids, _ = stag.detectMarkers(image, **config)

    if ids is None:
        print("Nenhum marcador foi encontrado.")
        return

    # Processar cada marcador encontrado
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)

        # Calcular a área do polígono do marcador
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        height = np.max(corner[:, 1]) - np.min(corner[:, 1])

        # Tamanho conhecido do marcador em milímetros
        marker_size_mm = 20  # mm
        pixel_size_mm = width / marker_size_mm  # Conversão de pixel para mm

        # Calcular o centróide do marcador
        M = cv2.moments(corner)
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
        else:
            centroid_x, centroid_y = 0, 0

        # Dimensões desejadas do crop em mm convertidas para pixels
        crop_width = int(75 * pixel_size_mm)  
        crop_height = int(25 * pixel_size_mm)  

        # Ajuste para subir a área de crop em 30 mm
        crop_y_adjustment = int(10 * pixel_size_mm)

        # Calcular a área de crop na imagem
        x_min = max(centroid_x - crop_height, 0)
        x_max = min(centroid_x + crop_height, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)  # Ajustado para subir a área de crop
        y_max = centroid_y - crop_y_adjustment  # Ajustado para subir a área de crop

        # Desenhar a área de crop na imagem
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 4)

        # Rotular o ID do marcador
        cv2.putText(image, f'ID: {id_}', (centroid_x - (-30), centroid_y - (-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Desenhar o contorno do marcador
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)

        # Desenhar um círculo vermelho no centróide
        cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

    # Mostrar a imagem com os contornos, centróides e a área de crop dos marcadores
    cv2.imshow('Marcadores STag', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Por favor, insira o caminho da imagem: ")
    detect_and_label_stags(image_path)
