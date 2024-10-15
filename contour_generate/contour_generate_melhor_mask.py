import cv2
import numpy as np
import pickle
import os
from rembg import remove
from PIL import Image

def load_image_and_remove_bg(image_path):
    """Carrega uma imagem colorida, remove fundo e processar imagem."""
    input_image = open(image_path, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3] 

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
    #TODO: extrair features para ORB, SIFT, SURF 
    orb = cv2.ORB_create()
    features = []
    for contour in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        keypoints, descriptors = orb.detectAndCompute(image, mask)
        if keypoints and descriptors is not None:
            features.append((keypoints, descriptors))
    return features

def transform_alpha_mask_png_to_white_255(image, contours):
    """Torna a máscara alpha visível aos olhos para ser tratada na detecção."""
    #TODO verificar erros ao executar quando acessa .convert e .new
    final_png = Image.convert(contours)
    background = Image.new('RGBA', final_png.size, (255,255,255))
    alpha_composite_off = final_png.alpha_composite_off(background, final_png)
    return alpha_composite_off
# im = PIL.Image.new(mode = "RGB", size = (200, 200),
# 						color = (153, 153, 255))

# # this will show image in any image viewer
# im.show()


def detect_and_save_features(mask, feature_type, directory, index):
    """recebe a imagem sem fundo processada em formato png com a máscara alpha em branco (255,255,255),
    limpa a máscara alpha excluindo esses valores e salva o contorno vetorizado, sem a referência da localização do contorno """
    if feature_type == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif feature_type == 'SURF':
        feature_detector = cv2.xfeatures2d.SURF_create()
    elif feature_type == 'ORB':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported feature type")

    keypoints, descriptors = feature_detector.detectAndCompute(mask, None)
    filename = f"contour_{feature_type}_{index}.pkl"

def save_features(features, directory="features"):
    """Salva os recursos de contorno SIFT, SURF, ORB (vetores sem orientação de localização na imagem) na pasta contour_saved com a informação contour+ORB+número_crescente"""
    #TODO salvar os contornos vetorizados em SIFT, SURF e ORB na pasta contour_saved
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
    image = load_image_and_remove_bg(image_path)
    mask = create_mask(image)
    img_cont, cont = extract_and_draw_contours(image, mask)
    transform_alpha= transform_alpha_mask_png_to_white_255(img_cont, cont)
    
    cv2.imshow("alpha",transform_alpha ) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    features = extract_features(image, cont)
    save_features(features)

if __name__ == "__main__":
    main()

