import cv2
import numpy as np
import stag  # Certifique-se de que a biblioteca stag está corretamente importada
import os

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

    # Dicionário para armazenar os dados de crop de cada marcador
    crop_data = {}

    # Preparar visualização de cada marcador encontrado
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)

        # Calcular o centróide do marcador
        M = cv2.moments(corner)
        centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        centroid_y = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
        cv2.putText(image, f'ID: {id_}', (centroid_x - 10, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Calcular área de crop
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20  # Tamanho conhecido do marcador em milímetros (20mm)
        crop_width = int(60 * pixel_size_mm)
        crop_height = int(20 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)
        x_min = max(centroid_x - crop_height, 0)
        x_max = min(centroid_x + crop_height, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = centroid_y - crop_y_adjustment
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)

        # Armazenar dados de crop para uso posterior
        crop_data[id_] = (x_min, x_max, y_min, y_max)

    # Mostrar a imagem com os contornos, centróides, IDs e áreas de crop
    cv2.imshow('Marcadores STag', image)
    cv2.waitKey(1)  # Apenas para renderizar a janela sem pausar aqui

    # Perguntar ao usuário qual ID ele deseja cortar
    input_id = input("Por favor, insira o ID do STag para a operação de crop: ")
    try:
        target_id = int(input_id)
        cv2.destroyAllWindows()  # Fechar a janela de visualização
    except ValueError:
        print(f"Entrada inválida: {input_id} não é um número válido.")
        cv2.destroyAllWindows()  # Fechar a janela de visualização
        return

    # Criar pasta 'crop' se não existir
    if not os.path.exists('crop'):
        os.makedirs('crop')

    # Processar o marcador escolhido para o crop
    if target_id in crop_data:
        x_min, x_max, y_min, y_max = crop_data[target_id]
        cropped_image = image[y_min:y_max, x_min:x_max]
        crop_filename = f'crop/{target_id}_crop.png'
        cv2.imwrite(crop_filename, cropped_image)
        print(f'Imagem cortada do marcador ID {target_id} salva em {crop_filename}')
    else:
        print(f"Marcador com ID {target_id} não encontrado.")

if __name__ == "__main__":
    image_path = input("Por favor, insira o caminho da imagem: ")
    detect_and_label_stags(image_path)
