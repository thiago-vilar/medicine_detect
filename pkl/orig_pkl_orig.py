import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_features(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    # Ajustar a reconstrução dos keypoints
    keypoints = [
        cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1], _angle=kp[2],
                     _response=kp[3], _octave=kp[4], _class_id=kp[5]) for kp in data['keypoints']
    ]
    # Assumindo que os descritores precisam ser reconstruídos em um array numpy
    descriptors = np.frombuffer(data['descriptors'], dtype=np.uint8)
    descriptors = descriptors.reshape((-1, 32))  # Ajustar o shape conforme necessário
    return keypoints, descriptors

def visualize_keypoints(image_path, keypoints):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(image, (x, y), int(kp.size), (0, 255, 0), 2)  # Desenha um círculo verde para cada keypoint

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title('Visualização dos Keypoints')
    plt.show()

def main():
    features_file = input("Digite o caminho do arquivo de features (.pkl): ")
    image_path = input("Digite o caminho da imagem original: ")
    keypoints, descriptors = load_features(features_file)
    visualize_keypoints(image_path, keypoints)

if __name__ == "__main__":
    main()
