import cv2
import numpy as np
from rembg import remove
import pickle
import os
import csv
from PIL import Image, ImageDraw

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Falha ao carregar a imagem.")
    return img

def extract_features(image, contours):
    orb = cv2.ORB_create()
    features = []
    for contour in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        keypoints, descriptors = orb.detectAndCompute(image, mask)
        features.append((keypoints, descriptors))
    return features

def convert_keypoints_to_data(keypoints):
    """Converte keypoints do OpenCV para um formato serializável."""
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def save_features(features, directory="features"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, "orb_features.pkl")
    with open(filename, 'wb') as f:
        # Salva os keypoints convertidos em uma forma que pode ser carregada posteriormente
        data = {'keypoints': [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in features.keypoints],
                'descriptors': features.descriptors}
        pickle.dump(data, f)


def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image(image_path)
    # Suponha que os contornos são extraídos aqui ou você tem uma função similar
    contours = [np.array([[[10, 10]], [[100, 100]], [[100, 10]]], dtype=np.int32)]  # Exemplo de contorno
    features = extract_features(image, contours)
    save_features(features)

if __name__ == "__main__":
    main()
