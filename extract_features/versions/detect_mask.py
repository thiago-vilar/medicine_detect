import matplotlib.pyplot as plt
import pickle
import numpy as np

def load_mask(file_path):
    """Carrega uma máscara de um arquivo .pkl."""
    with open(file_path, 'rb') as file:
        mask = pickle.load(file)
    return mask

def display_masks(mask1, mask2, title1='Mask 1', title2='Mask 2'):
    """Exibe duas máscaras lado a lado para comparação."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mask1, cmap='gray', interpolation='none')
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask2, cmap='gray', interpolation='none')
    plt.title(title2)
    plt.axis('off')
    plt.show()

def calculate_iou(mask1, mask2):
    """Calcula a Intersecção sobre União (IoU) entre duas máscaras."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Caminhos para os arquivos .pkl
mask_file_path = 'features\mask\mask_0.pkl'
#TODO detectar máscara e comparar com máscara carregada com 
real_mask_file_path = ''  

# Carrega as máscaras
predicted_mask = load_mask(mask_file_path)
real_mask = load_mask(real_mask_file_path)

# Exibe as máscaras lado a lado
display_masks(predicted_mask, real_mask, title1='Predicted Mask', title2='Real Mask')

# Calcula e imprime a IoU
iou = calculate_iou(predicted_mask, real_mask)
print(f"The Intersection over Union (IoU) is: {iou:.4f}")
