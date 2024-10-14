import cv2
import numpy as np
import pickle
import os
from rembg import remove

def load_image(image_path):
    """Carrega uma imagem colorida e processa para extração de contornos."""
    input_image = open(image_path, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_mask(img):
    """Cria uma máscara para os contornos com base em um limiar de cor."""
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    mask = cv2.inRange(img, lower_bound, upper_bound)
    return mask

def extract_and_draw_contours(img, mask):
    """Extrai contornos da máscara e desenha sobre a imagem."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
    return img_with_contours, contours

def extract_features(image, contours):
    """Extrai características usando o ORB para os contornos dados."""
    orb = cv2.ORB_create()
    features = []
    for contour in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        keypoints, descriptors = orb.detectAndCompute(image, mask)
        if keypoints and descriptors is not None:
            features.append((keypoints, descriptors))
    return features

def save_features(features, directory="features"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, (keypoints, descriptors) in enumerate(features):
        data = {
            'keypoints': [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints],
            'descriptors': descriptors.tobytes()
        }
        filename = os.path.join(directory, f"features{i}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Features saved in {filename}")

def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image(image_path)
    mask = create_mask(image)
    img_with_contours, contours = extract_and_draw_contours(image, mask)
    cv2.imshow("Contornos Detectados", img_with_contours)  # Mostra os contornos na imagem
    cv2.waitKey(0)  # Aguarda pressionar qualquer tecla
    cv2.destroyAllWindows()  # Fecha a janela aberta
    features = extract_features(image, contours)
    save_features(features)

if __name__ == "__main__":
    main()
