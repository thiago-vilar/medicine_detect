import pickle
import matplotlib.pyplot as plt

def load_and_plot_signature(filename):
    with open(filename, 'rb') as f:
        contour_shifted = pickle.load(f)
    
    # Plotando para verificar se os pontos estão corretos
    plt.figure()
    plt.scatter(contour_shifted[:, 0], contour_shifted[:, 1], c='green', s=10)  # Usar scatter garante que não há conexões
    plt.axis('off')
    plt.show()

# Exemplo de uso
load_and_plot_signature('.\\signature_contours\\20241004090606.pkl')
