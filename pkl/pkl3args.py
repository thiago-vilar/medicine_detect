import cv2
import numpy as np
import pickle
import sys

def load_features(features_path):
    """Carrega os keypoints e descriptors de um arquivo .pkl."""
    with open(features_path, 'rb') as file:
        data = pickle.load(file)
    # Corrigindo a reconstrução de KeyPoint para garantir que todos os argumentos estão corretos
    keypoints = [
        cv2.KeyPoint(
            x=float(kp[0][0]), 
            y=float(kp[0][1]), 
            _size=float(kp[1]), 
            _angle=float(kp[2]), 
            _response=float(kp[3]), 
            _octave=int(kp[4]), 
            _class_id=int(kp[5])
        ) for kp in data['keypoints']
    ]
    descriptors = np.frombuffer(data['descriptors'], dtype=np.uint8)
    descriptors = descriptors.reshape((-1, 32))  # Ajuste conforme o número de colunas dos descriptors usados
    return keypoints, descriptors

def main():
    if len(sys.argv) > 1:
        features_path = sys.argv[1]
    else:
        features_path = input("Digite o caminho para o arquivo de features (.pkl): ")
    
    try:
        keypoints, descriptors = load_features(features_path)
        print("Features carregadas com sucesso.")
        print(f"Total de Keypoints carregados: {len(keypoints)}")
    except Exception as e:
        print(f"Erro ao carregar features: {e}")

if __name__ == "__main__":
    main()
