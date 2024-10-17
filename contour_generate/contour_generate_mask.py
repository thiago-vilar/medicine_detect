import cv2
import numpy as np
import pickle
import os
from rembg import remove

def load_image_and_remove_bg(image_path):
    """Carrega uma imagem e remove o fundo usando a biblioteca rembg."""
    with open(image_path, 'rb') as file:
        input_image = file.read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    return img if img is not None else None  # Preservar o canal alpha

def create_mask(img):
    """Cria uma máscara binária para detecção de contornos baseada em limiares de cor."""
    lower_bound = np.array([30, 30, 30, 0])  # Inclui o canal alpha
    upper_bound = np.array([250, 250, 250, 255])
    return cv2.inRange(img, lower_bound, upper_bound)

def find_centroid(mask):
    """Encontra o centróide da máscara."""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(round(M["m10"] / M["m00"]))
        cY = int(round(M["m01"] / M["m00"]))
        return cX, cY
    else:
        return None

def extract_and_draw_contours(img, mask):
    """Extrai contornos da máscara e desenha-os sobre a imagem, incluindo o centróide."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = img.copy()
    if img_with_contours.shape[2] == 3:
        img_with_contours = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2BGRA)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0, 255), 1)
    centroid = find_centroid(mask)
    if centroid:
        cX, cY = centroid
        if 0 <= cY < img_with_contours.shape[0] and 0 <= cX < img_with_contours.shape[1]:
            cv2.circle(img_with_contours, (cX, cY), 3, (0, 0, 255, 255), -1)  # Desenhar o centróide em vermelho
    return img_with_contours, contours, centroid

def crop_to_contours(img, contours):
    """Recorta a imagem para a bounding box mínima que envolve todos os contornos."""
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        return img[y:y+h, x:x+w]
    return img

def transform_image_alpha_to_white_background(img_cv):
    """Remove o canal alpha de uma imagem, substituindo-o por um fundo branco."""
    if img_cv.shape[2] == 4:  # Imagem com canal alpha
        alpha_channel = img_cv[:, :, 3]
        rgb_channels = img_cv[:, :, :3]
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        img_with_background = rgb_channels * alpha_factor + white_background_image * (1 - alpha_factor)
        img_with_background = img_with_background.astype(np.uint8)
    else:
        img_with_background = img_cv
    return img_with_background

def extract_features(img, mask, centroid):
    """Extrai características (keypoints e descritores) da imagem usando a máscara para limitar a área."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    features = {}
    if keypoints and descriptors is not None:
        if centroid:
            cX, cY = centroid
            # Ajustar as coordenadas dos keypoints para que o centróide esteja em (0,0)
            adjusted_keypoints = [
                cv2.KeyPoint(kp.pt[0] - cX, kp.pt[1] - cY, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in keypoints
            ]
            features['orb'] = (adjusted_keypoints, descriptors)
        else:
            features['orb'] = (keypoints, descriptors)
    else:
        print("Nenhuma característica foi encontrada.")
    return features

def save_features(features, contours, centroid, directory="features"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not features:
        print("Nenhuma característica para salvar.")
        return
    for feature_type, (keypoints, descriptors) in features.items():
        keypoints_serializable = []
        for kp in keypoints:
            temp = {
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            }
            keypoints_serializable.append(temp)
        feature_data = {
            'keypoints': keypoints_serializable,
            'descriptors': descriptors.tolist(),
            'contours': [contour.tolist() for contour in contours],
            'centroid': centroid
        }
        filename = os.path.join(directory, f"{feature_type}_features.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(feature_data, f)
        print(f"Características {feature_type} salvas em {filename}")

def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image_and_remove_bg(image_path)
    if image is None:
        print("Erro ao processar a imagem.")
        return

    # Criar a máscara e extrair os contornos
    mask = create_mask(image)
    img_cont, contours, centroid = extract_and_draw_contours(image, mask)
    img_final = transform_image_alpha_to_white_background(img_cont)
    img_cropped = crop_to_contours(img_final, contours)

    # Salvar a imagem recortada com contornos em um fundo branco
    cv2.imwrite('contoured_cropped_image_output.png', img_cropped)
    print("Imagem recortada com contornos salva em fundo branco: contoured_cropped_image_output.png")

    # Extrair as características e salvar junto com os contornos e o centróide
    features = extract_features(image, mask, centroid)
    save_features(features, contours, centroid)

if __name__ == "__main__":
    main()
