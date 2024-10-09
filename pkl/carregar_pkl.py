import pickle
import matplotlib.pyplot as plt
import numpy as np  # Importando numpy para manipulação de arrays, se necessário

def load_and_plot_signature(filename):
    with open(filename, 'rb') as f:
        contour = pickle.load(f)

    # Verificar se os pontos estão na forma correta e ajustar se necessário
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = contour.squeeze(axis=1)  # Ajustar para (N, 2)

    # Plotando para verificar se os pontos estão corretos
    plt.figure()
    plt.scatter(contour[:, 0], contour[:, 1], c='green', s=10)  # Usar scatter garante que não há conexões
    plt.axis('equal')  # Ajusta o aspecto para que 1 unidade no eixo x seja igual a 1 unidade no eixo y
    plt.axis('off')  # Oculta os eixos para uma visualização limpa
    plt.show()

# Exemplo de uso
load_and_plot_signature('contours/full_contour_2.pkl')
