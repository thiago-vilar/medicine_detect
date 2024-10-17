import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_features(feature_file):
    """Carrega as características ORB salvas em um arquivo pickle."""
    with open(feature_file, 'rb') as f:
        feature_data = pickle.load(f)
    return feature_data

def plot_features(features):
    """Plota os keypoints ORB e o centróide em um gráfico de dispersão."""
    keypoints = features['keypoints']
    contours = features.get('contours', [])
    centroid = features.get('centroid', None)

    # Extrair as coordenadas dos keypoints
    x_coords = [kp['pt'][0] for kp in keypoints]
    y_coords = [kp['pt'][1] for kp in keypoints]

    # Plotar os keypoints
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, c='red', marker='o', label='Keypoints ORB')

    # Plotar os contornos, se disponíveis
    for contour in contours:
        contour = np.array(contour)
        if contour.shape[1] == 2:  # Verificar se o contorno tem pelo menos duas dimensões
            plt.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=1)

    # Plotar o centróide no ponto (0,0), se disponível
    if centroid:
        plt.scatter(0, 0, c='green', marker='x', s=100, label='Centróide (0,0)')

    plt.title("Características ORB Salvas")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.gca().invert_yaxis()  
    plt.legend()
    plt.show()

def main():
    feature_file = input("Digite o caminho para o arquivo orb_features.pkl: ")
    features = load_features(feature_file)
    plot_features(features)

if __name__ == "__main__":
    main()
