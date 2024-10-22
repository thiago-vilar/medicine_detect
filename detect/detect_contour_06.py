import cv2
import numpy as np
import pickle
import os

def load_image(image_path):
    """Carrega uma imagem colorida."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem não pode ser carregada. Verifique o caminho do arquivo.")
    return img

def load_features(features_path):
    """Carrega features de um arquivo .pkl."""
    with open(features_path, 'rb') as file:
        data = pickle.load(file)
    return data['keypoints'], data['descriptors']

def convert_keypoints_from_data(keypoint_data):
    """Converte dados de keypoints serializados de volta para objetos KeyPoint do OpenCV."""
    # Assegura que os argumentos estão sendo passados na ordem e forma correta
    return [cv2.KeyPoint(x=float(kp[0][0]), y=float(kp[0][1]), _size=float(kp[1]), 
                         _angle=float(kp[2]), _response=float(kp[3]), _octave=int(kp[4]), 
                         _class_id=int(kp[5])) for kp in keypoint_data]

def detect_and_draw_contours(image, keypoints, descriptors):
    """Detecta e desenha contornos na imagem usando keypoints e descritores."""
    orb = cv2.ORB_create()
    kp2, des2 = orb.detectAndCompute(image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    matched_image = cv2.drawMatchesKnn(image, keypoints, image, kp2, [[m] for m in matches[:10]], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

def save_image(image, directory="saved", base_filename="contour_detected"):
    """Salva a imagem em um diretório específico com numeração sequencial."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    i = 0
    while os.path.exists(os.path.join(directory, f"{base_filename}{i}.png")):
        i += 1
    filename = os.path.join(directory, f"{base_filename}{i}.png")
    cv2.imwrite(filename, image)
    return filename

def main():
    image_path = input("Digite o caminho da imagem: ")
    features_path = input("Digite o caminho do arquivo de features (.pkl): ")

    image = load_image(image_path)
    keypoint_data, descriptors = load_features(features_path)
    keypoints = convert_keypoints_from_data(keypoint_data)
    
    result_image = detect_and_draw_contours(image, keypoints, descriptors)

    saved_path = save_image(result_image)
    print(f"Imagem salva em: {saved_path}")

    cv2.imshow("Imagem com Contorno Especificado", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
