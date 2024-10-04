import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import pickle
import os
import stag

def remove_background(filepath):
    input_image = open(filepath, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    return img[:, :, :3] if img is not None else None

def detect_stags(img):
    # Supondo que a biblioteca 'stag' possa ser usada de forma semelhante
    detector = STag()
    markers = detector.detect(img)
    return markers

def display_markers(img, markers):
    for marker in markers:
        cv2.polylines(img, [marker.corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(img, str(marker.id), tuple(marker.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected STags")
    plt.show()

def read_contour_signature(filename):
    with open(filename, 'rb') as f:
        contour = pickle.load(f)
    return contour

def match_contour(img, input_contour):
    _, contours, _ = cv2.findContours(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        similarity = cv2.matchShapes(input_contour, contour, 1, 0.0)
        if similarity < 0.1:  # Assumindo um limiar de similaridade
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 3)
            return True, similarity
    return False, None

def main():
    filepath = input("Digite o caminho da imagem: ")
    img = remove_background(filepath)
    if img is None:
        print("Falha ao processar a imagem.")
        return

    markers = detect_stags(img)
    display_markers(img.copy(), markers)

    selected_id = int(input("Digite o ID do STag: "))
    for marker in markers:
        if marker.id == selected_id:
            print(f"Área do STag {selected_id}: {cv2.contourArea(marker.corners)}")

    signature_path = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    input_contour = read_contour_signature(signature_path)

    match_found, similarity = match_contour(img, input_contour)
    if match_found:
        print(f"Match encontrado! Similaridade: {similarity:.2f}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Match de Contorno Encontrado")
        plt.show()
    else:
        print("Nenhum match encontrado.")

if __name__ == "__main__":
    main()
