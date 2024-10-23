import cv2
import numpy as np
import pickle
import os

def load_contour_features(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if 'contour' in data and 'features' in data:
            return data['contour'], data['features']
        else:
            print("O arquivo não contém os dados esperados ('contour' ou 'features').")
            return None, None
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {filename}")
        return None, None
    except Exception as e:
        print(f"Erro ao carregar dados do arquivo {filename}: {e}")
        return None, None

def match_contours(contour, features, image, threshold=10):
    if contour is None or features is None:
        print("Dados de contorno ou características não disponíveis.")
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [np.array(contour)], -1, 255, -1)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, mask)

    if descriptors is None:
        print("Nenhum descritor encontrado na imagem nova.")
        return None

    if 'SIFT' not in features or 'descriptors' not in features['SIFT']:
        print("Dados de características SIFT ausentes ou inválidos no arquivo.")
        return None

    saved_descriptors = features['SIFT']['descriptors']
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(saved_descriptors, descriptors, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches) > threshold:
        return cv2.drawContours(image.copy(), [np.array(contour)], -1, (0, 255, 0), 3)
    return None

def process_new_image(image_path, contour_file):
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    contour, features = load_contour_features(contour_file)
    if contour is None or features is None:
        return

    matched_image = match_contours(contour, features, image)
    if matched_image is not None:
        cv2.imshow("Matched Image", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhum contorno correspondente encontrado.")

if __name__ == "__main__":
    image_path = input("Digite o caminho da nova imagem: ")
    contour_file = input("Digite o caminho do arquivo de contorno: ")
    process_new_image(image_path, contour_file)
