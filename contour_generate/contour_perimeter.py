import cv2
import numpy as np
import pickle
import os
from rembg import remove
from PIL import Image

def load_image_and_remove_bg(image_path):
    """Carrega uma imagem e remove o fundo usando a biblioteca rembg."""
    try:
        with open(image_path, 'rb') as file:
            input_image = file.read()
        output_image = remove(input_image)
        image_np_array = np.frombuffer(output_image, np.uint8)
        img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
        return img[:, :, :3] if img is not None else None
    except Exception as e:
        print(f"Erro ao carregar ou processar a imagem: {e}")
        return None

def create_mask(img):
    """Cria uma máscara binária para detecção de contornos baseada em limiares de cor."""
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)


def extract_contours(mask):
    """Extrai contornos da máscara fornecida e retorna aqueles com área suficiente."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [] 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area == True:
            peri = cv2.arcLength(cnt, True)
            approx_contour = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            significant_contours.append(approx_contour)  
            cv2.drawContours(mask, [approx_contour], -1, (255, 0, 255), 7)
            print(approx_contour)
    return significant_contours, hierarchy

def draw_contours(img, contours):
    """Desenha contornos sobre a imagem fornecida."""
    if contours:  
        return cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 1)
    return img

def save_contours_and_features(contours, image, directory="contours_features"):
    """Salva os contornos e as características associadas para reconhecimento futuro."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    feature_detectors = {'ORB': cv2.ORB_create(), 'SIFT': cv2.SIFT_create()}
    for index, contour in enumerate(contours):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        feature_data = {}

        for key, detector in feature_detectors.items():
            keypoints, descriptors = detector.detectAndCompute(image, mask)
            if keypoints:
                keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
                feature_data[key] = {'keypoints': keypoints_serializable, 'descriptors': descriptors}

        contour_data = {'contour': contour.tolist(), 'features': feature_data}
        filename = os.path.join(directory, f"contour_{index}_features.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(contour_data, f)
        print(f"Contour and features saved in {filename}")



def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image_and_remove_bg(image_path)
    if image is None:
        print("Erro ao processar a imagem. Encerrando...")
        return

    mask = create_mask(image)
    contours, hierarchy = extract_contours(mask)
    if not contours:  # Verifica se nenhum contorno significativo foi encontrado
        print("Nenhum contorno significativo encontrado.")
        return

    img_with_contours = draw_contours(image, contours)  # Passa a lista completa de contornos
    save_contours_and_features(contours, image)  

    cv2.imwrite('contoured_image_output_epsilon.png', img_with_contours)
    print("Imagem com contornos salvos: contoured_image_output_epsilon.png")

if __name__ == "__main__":
    main()
