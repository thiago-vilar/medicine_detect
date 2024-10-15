import cv2
import numpy as np
import pickle
import os
from rembg import remove
from PIL import Image

def load_image_and_remove_bg(image_path):
    input_image = open(image_path, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    return img[:, :, :3]

def create_mask(img):
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    mask = cv2.inRange(img, lower_bound, upper_bound)
    return mask

def extract_and_draw_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 3)
    return img_with_contours, contours

def transform_image_alpha_to_white_background(img_cv):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    if img_pil.mode == 'RGBA':
        background = Image.new('RGBA', img_pil.size, (255, 255, 255, 255))
        img_pil = Image.alpha_composite(background, img_pil)
    return cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

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
    """Salva os recursos de contorno (ORB e SIFT) em arquivos separados."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    for feature_type, data in features.items():
        keypoints, descriptors = data
        # Converte keypoints para algo que possa ser serializado
        keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
        feature_data = {'keypoints': keypoints_serializable, 'descriptors': descriptors}
        filename = os.path.join(directory, f"{feature_type}_features.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(feature_data, f)
        print(f"{feature_type} features saved in {filename}")


def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image_and_remove_bg(image_path)
    mask = create_mask(image)
    img_cont, cont = extract_and_draw_contours(image, mask)
    img_final = transform_image_alpha_to_white_background(img_cont)
    cv2.imwrite('contoured_image_output.png', img_final)
    print("Imagem com contornos salvos em fundo branco: contoured_image_output.png")
    features = extract_features(image, cont)
    save_features(features)

if __name__ == "__main__":
    main()
