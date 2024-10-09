import cv2
import numpy as np
import stag
import pickle
from datetime import datetime
import os

def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None or color_image is None:
        print("Erro ao carregar a imagem.")
        return None, None, None, None

    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    corners, ids, _ = stag.detectMarkers(color_image, **config)
    if ids is None or len(ids) == 0:
        print("Nenhum marcador foi encontrado.")
        return None, None, image, color_image

    return corners, ids, image, color_image

def display_markers(image, corners, ids):
    scan_areas = {}
    if corners is None or ids is None:
        return scan_areas

    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x, centroid_y = int(np.mean(corner[:, 0])), int(np.mean(corner[:, 1]))
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)
        cv2.putText(image, f'ID: {id_}', (centroid_x, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

        # Cálculo da 'Scan Area' com base no tamanho do marcador
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20  # Supondo que o marcador tenha 20 mm de largura
        crop_width = int(75 * pixel_size_mm)
        crop_height = int(50 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        x_min = max(centroid_x - crop_height // 2, 0)
        x_max = min(centroid_x + crop_height // 2, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = min(centroid_y - crop_y_adjustment, image.shape[0])

        scan_area = ((x_min, y_min), (x_max, y_max))
        scan_areas[id_] = scan_area

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        cv2.putText(image, 'Scan Area', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return scan_areas

def read_contour_signature(filename):
    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado.")
        return None
    with open(filename, 'rb') as file:
        return pickle.load(file)

def create_contour_signature(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erro ao carregar a imagem de referência.")
        return False
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        # Supondo que o maior contorno é o objeto de interesse
        contour = max(contours, key=cv2.contourArea)
        with open(output_path, 'wb') as file:
            pickle.dump(contour, file)
        print(f"Assinatura do contorno salva em {output_path}")
        return True
    else:
        print("Nenhum contorno encontrado na imagem de referência.")
        return False

def match_contour_in_area(image, scan_area, contour_signature, similarity_threshold=0.5):
    if contour_signature is None:
        print("Assinatura do contorno não foi carregada.")
        return False, None

    x_min, y_min = scan_area[0]
    x_max, y_max = scan_area[1]
    region_of_interest = image[y_min:y_max, x_min:x_max]

    # Verificar se a região de interesse é colorida ou em escala de cinza
    if len(region_of_interest.shape) == 3:
        gray_roi = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = region_of_interest.copy()

    # Aplicar limiarização para melhorar a detecção de contornos
    _, thresh = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)

    # Detectar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best_match = None
    min_similarity = float('inf')
    for test_contour in contours:
        similarity = cv2.matchShapes(contour_signature, test_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < min_similarity:
            min_similarity = similarity
            best_match = test_contour

    if best_match is not None and min_similarity <= similarity_threshold:
        # Ajustar coordenadas do contorno para o contexto da imagem completa
        full_image_contour = best_match.copy()
        full_image_contour[:, 0, 0] += x_min
        full_image_contour[:, 0, 1] += y_min

        # Usar a imagem colorida original para desenhar o contorno
        result_image = image.copy()

        # Desenhar o contorno correspondente na imagem em vermelho
        cv2.drawContours(result_image, [full_image_contour.astype(int)], -1, (0, 0, 255), 2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_contour_{timestamp}.jpg"

        # Salvar a imagem
        cv2.imwrite(filename, result_image)
        print(f"Match encontrado! Similaridade aproximada: {100 - min_similarity * 100:.2f}%")
        print(f"Imagem salva como {filename}")
        cv2.imshow('Match de Contorno Encontrado', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True, min_similarity

    print(f"Nenhum match encontrado com similaridade superior a {100 - similarity_threshold * 100}%.")
    return False, None


def main():
    image_path = input("Por favor, insira o caminho da imagem: ")
    corners, ids, image, color_image = detect_and_label_stags(image_path)
    if image is not None and ids is not None:
        scan_areas = display_markers(image, corners, ids)
        cv2.imshow('Marcadores STag', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Marcadores detectados: {list(scan_areas.keys())}")
        selected_id = int(input("Digite o ID do STag para a área de varredura: "))
        if selected_id in scan_areas:
            signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
            input_contour = read_contour_signature(signature_path)
            if input_contour is None:
                create_new = input("Arquivo de assinatura do contorno não encontrado. Deseja criar um novo? (s/n): ").lower()
                if create_new == 's':
                    reference_image_path = input("Insira o caminho da imagem de referência para extrair o contorno: ")
                    if create_contour_signature(reference_image_path, signature_path):
                        input_contour = read_contour_signature(signature_path)
                    else:
                        print("Não foi possível criar a assinatura do contorno.")
                        return
                else:
                    print("Assinatura do contorno não fornecida. Encerrando o programa.")
                    return
            scan_area = scan_areas[selected_id]
            print(f"Processando Scan Area do marcador ID {selected_id}")

            # Solicitar o limiar de similaridade do usuário (opcional)
            similarity_input = input("Digite o percentual mínimo de similaridade desejado (0-100, padrão 80): ")
            if similarity_input.strip() == '':
                similarity_threshold = 0.5  # Padrão 80% de similaridade
            else:
                similarity_percent = float(similarity_input)
                similarity_threshold = (100 - similarity_percent) / 100  # Converter para a métrica da função

            match_found, similarity = match_contour_in_area(color_image, scan_area, input_contour, similarity_threshold)
            if match_found:
                print(f"Match encontrado no marcador ID {selected_id}! Similaridade aproximada: {100 - similarity * 100:.2f}%")
            else:
                print(f"Nenhum match encontrado no marcador ID {selected_id}.")
        else:
            print(f"O marcador ID {selected_id} não foi detectado na imagem.")
    else:
        print("Não foi possível processar a imagem.")

if __name__ == "__main__":
    main()
