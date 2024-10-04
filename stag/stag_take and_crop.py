import cv2
import numpy as np
import stag
import os
import threading
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

# Variável global para armazenar os IDs de entrada
input_ids = None

def thread_input():
    global input_ids
    input_str = input("Por favor, insira o(s) ID(s) do STag para a operação de crop (separados por vírgula): ")
    input_ids = [int(id_.strip()) for id_ in input_str.split(',')]


def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Configurar a detecção de marcadores STag
    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    # Detectar marcadores na imagem
    corners, ids, _ = stag.detectMarkers(image, **config)

    if ids is None:
        print("Nenhum marcador foi encontrado.")
        return

    # Dicionário para armazenar os dados de crop de cada marcador
    crop_data = {}

    # Preparar visualização de cada marcador encontrado
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)

        # Calcular o centróide do marcador
        M = cv2.moments(corner)
        centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        centroid_y = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
        cv2.putText(image, f'ID: {id_}', (centroid_x + 30, centroid_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Calcular área de crop
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20  
        crop_width = int(60 * pixel_size_mm)
        crop_height = int(20 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)
        x_min = max(centroid_x - crop_height, 0)
        x_max = min(centroid_x + crop_height, image.shape[1])
        y_min = max(centroid_y - crop_width - crop_y_adjustment, 0)
        y_max = centroid_y - crop_y_adjustment
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        # Armazenar dados de crop para uso posterior
        crop_data[id_] = (x_min, x_max, y_min, y_max)

    # Usar tkinter para criar uma janela sempre no topo
    root = tk.Tk()
    root.title("Marcadores STag")
    # Definir a janela como sempre no topo
    root.attributes('-topmost', True)

    # Converter a imagem do OpenCV para o formato do PIL
    cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    image_tk = ImageTk.PhotoImage(image=pil_image)

    label_image = tk.Label(root, image=image_tk)
    label_image.pack()

    # Executar thread de entrada enquanto a imagem é exibida
    input_thread = threading.Thread(target=thread_input)
    input_thread.start()

    # Mostrar janela tkinter
    root.mainloop()
    input_thread.join()  # Garantir que a thread de entrada conclua antes de continuar

    # Fechar a janela tkinter ao receber os IDs
    root.quit()

    # Processar entrada e crop conforme necessário
    if input_ids:
        # Criar pasta 'crop' se não existir
        if not os.path.exists('crop'):
            os.makedirs('crop')

        # Processar os IDs fornecidos
        for target_id in input_ids:
            if target_id in crop_data:
                x_min, x_max, y_min, y_max = crop_data[target_id]
                cropped_image = image[y_min:y_max, x_min:x_max]
                crop_filename = f'crop/{target_id}_crop.png'
                cv2.imwrite(crop_filename, cropped_image)
                print(f'Imagem cortada do marcador ID {target_id} salva em {crop_filename}')
            else:
                print(f"Marcador com ID {target_id} não encontrado.")

if __name__ == "__main__":
    image_path = input("Por favor, insira o caminho da imagem: ")
    detect_and_label_stags(image_path)
