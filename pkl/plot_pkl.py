import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_features(filename):
    """Carrega características de um arquivo .pkl e trata variações no formato dos dados."""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    keypoints = []
    if 'keypoints' in data and 'descriptors' in data:
        for kp in data['keypoints']:
            # Certifique-se de que todos os valores necessários estão disponíveis e são válidos
            x = kp[0][0] if len(kp[0]) > 0 else 0
            y = kp[0][1] if len(kp[0]) > 1 else 0
            size = kp[1] if len(kp) > 1 and kp[1] > 0 else 1  # size deve ser maior que 0
            angle = kp[2] if len(kp) > 2 else 0
            response = kp[3] if len(kp) > 3 else 0
            octave = kp[4] if len(kp) > 4 else 0
            class_id = kp[5] if len(kp) > 5 else -1
            keypoints.append(cv2.KeyPoint(x=x, y=y, _size=size, _angle=angle,
                                          _response=response, _octave=octave, _class_id=class_id))
        descriptors = np.array(data['descriptors'])
    else:
        print("Dados de keypoints ou descritores não encontrados no arquivo.")
        return None, None
    return keypoints, descriptors

def display_keypoints(keypoints):
    """Exibe keypoints em uma imagem em branco usando Matplotlib."""
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(canvas, (x, y), int(kp.size/2), (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints')
    plt.show()

def main():
    filename = input("Digite o caminho completo do arquivo .pkl: ")
    keypoints, descriptors = load_features(filename)
    if keypoints and descriptors:
        print(f"Número de vetores (keypoints): {len(keypoints)}")
        print(f"Shape dos descritores: {descriptors.shape}")
        print(f"Os vetores contêm informações de localização dos contornos na imagem: {'Sim' if keypoints else 'Não'}")
        display_keypoints(keypoints)
    else:
        print("Erro ao carregar dados.")

if __name__ == "__main__":
    main()
