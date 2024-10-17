import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_features(feature_file):
    """Carrega as características ORB salvas em um arquivo pickle."""
    with open(feature_file, 'rb') as f:
        feature_data = pickle.load(f)
    return feature_data

def match_features(descriptors1, descriptors2):
    """Encontra correspondências entre dois conjuntos de descritores usando BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    # Ordenar as correspondências pela distância (quanto menor a distância, melhor a correspondência)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def scan_image_for_match(template_features, search_image):
    """Percorre a nova imagem em busca de correspondências com base no centróide dos keypoints ORB."""
    keypoints_template = template_features['keypoints']
    descriptors_template = np.asarray(template_features['descriptors'], dtype=np.uint8)
    centroid = template_features['centroid']

    if centroid is None:
        print("Centróide não encontrado no arquivo de características.")
        return

    # Carregar e converter a nova imagem para tons de cinza
    gray_search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    found_match = False

    # Realizar a varredura da imagem para tentar encontrar uma correspondência
    step_size = 10  # Tamanho do passo da varredura (ajustar conforme necessário)
    for y in range(0, gray_search_image.shape[0], step_size):
        for x in range(0, gray_search_image.shape[1], step_size):
            # Definir uma região de interesse (ROI) centrada no ponto (x, y)
            roi_x1 = max(0, x - centroid[0])
            roi_y1 = max(0, y - centroid[1])
            roi_x2 = min(gray_search_image.shape[1], x + centroid[0])
            roi_y2 = min(gray_search_image.shape[0], y + centroid[1])
            roi = gray_search_image[roi_y1:roi_y2, roi_x1:roi_x2]

            # Extrair keypoints e descritores da região de interesse
            keypoints_roi, descriptors_roi = orb.detectAndCompute(roi, None)
            if descriptors_roi is None or len(keypoints_roi) == 0:
                continue

            # Tentar encontrar correspondências entre o template e a ROI
            matches = match_features(descriptors_template, descriptors_roi)

            # Considerar uma correspondência válida se houver uma quantidade suficiente de boas correspondências
            if len(matches) > 0.8 * len(descriptors_template):
                found_match = True
                print(f"Correspondência encontrada na posição: ({x}, {y})")

                # Desenhar a correspondência na imagem original
                cv2.rectangle(search_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
                break
        if found_match:
            break

    if found_match:
        # Mostrar a imagem com a região correspondente rotulada
        cv2.imshow("Resultado da Correspondência", search_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhuma correspondência encontrada.")

def main():
    feature_file = input("Digite o caminho para o arquivo orb_features.pkl: ")
    search_image_path = input("Digite o caminho da imagem para buscar correspondências: ")

    # Carregar as características ORB do arquivo
    template_features = load_features(feature_file)

    # Carregar a nova imagem onde a correspondência será buscada
    search_image = cv2.imread(search_image_path)
    if search_image is None:
        print("Erro ao carregar a imagem de busca.")
        return

    # Realizar a varredura na nova imagem para encontrar correspondências
    scan_image_for_match(template_features, search_image)

if __name__ == "__main__":
    main()
