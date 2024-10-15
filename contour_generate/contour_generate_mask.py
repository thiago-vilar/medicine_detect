import cv2
import numpy as np
import pickle
import os
from rembg import remove
from PIL import Image

def load_image_and_remove_bg(image_path):
    """Carrega uma imagem e remove o fundo usando a biblioteca rembg."""
    with open(image_path, 'rb') as file:
        input_image = file.read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    return img[:, :, :3] if img is not None else None

def create_mask(img):
    """Cria uma máscara binária para detecção de contornos baseada em limiares de cor."""
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_and_draw_contours(img, mask):
    """Extrai contornos da máscara e desenha-os sobre a imagem."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
    return img_with_contours, contours

def transform_image_alpha_to_white_background(img_cv):
    """Remove o canal alpha de uma imagem, substituindo-o por um fundo branco."""
    if img_cv.shape[2] == 4:  # Imagem com canal alpha
        alpha_channel = img_cv[:, :, 3]
        rgb_channels = img_cv[:, :, :3]
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        mask = alpha_channel[:, :, np.newaxis].astype(np.uint8) // 255
        base = white_background_image * (1 - mask)
        img_with_background = base + rgb_channels * mask
    else:
        img_with_background = img_cv
    return img_with_background

def crop_to_contours(img, contours):
    """Recorta a imagem para a bounding box mínima que envolve todos os contornos."""
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        return img[y:y+h, x:x+w]
    return img

def extract_features(image, contours):
    feature_detectors = {'ORB': cv2.ORB_create()}
    try:
        feature_detectors['SIFT'] = cv2.SIFT_create()
    except Exception as e:
        print("SIFT not available:", e)
    features = {}
    for key, detector in feature_detectors.items():
        for contour in contours:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            keypoints, descriptors = detector.detectAndCompute(image, mask)
            if keypoints:
                features[key] = (keypoints, descriptors)
    return features

def save_features(features, directory="features"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for feature_type, (keypoints, descriptors) in features.items():
        keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
        feature_data = {'keypoints': keypoints_serializable, 'descriptors': descriptors.tobytes()}
        filename = os.path.join(directory, f"{feature_type}_features.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(feature_data, f)
        print(f"{feature_type} features saved in {filename}")

def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image_and_remove_bg(image_path)
    if image is None:
        print("Erro ao processar a imagem.")
        return
    mask = create_mask(image)
    img_cont, cont = extract_and_draw_contours(image, mask)
    img_final = transform_image_alpha_to_white_background(img_cont)
    img_cropped = crop_to_contours(img_final, cont)
    
    cv2.imwrite('contoured_cropped_image_output.png', img_cropped)
    print("Imagem recortada com contornos salvos em fundo branco: contoured_cropped_image_output.png")
    
    features = extract_features(image, cont)
    save_features(features)

if __name__ == "__main__":
    main()
