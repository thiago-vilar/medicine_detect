import cv2
import stag
import numpy as np

def detect_stag_markers(image_path, library_hd=17, error_correction=None):
    """
    Detecta marcadores STag em uma imagem utilizando a biblioteca stag-python.

    Args:
    image_path (str): Caminho para a imagem onde os marcadores STag serão detectados.
    library_hd (int): Identifica a "família" ou "tipo" de marcadores STag a serem usados.
    error_correction (int): Quantidade de correção de erros, deve estar dentro de 0 <= error_correction <= (library_hd-1)/2.
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Define os parâmetros de configuração, se fornecidos
    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    # Detecta os marcadores na imagem
    (corners, ids, rejected_corners) = stag.detectMarkers(image, **config)

    # Verifica se algum marcador foi encontrado e os exibe
    if ids is not None and len(ids) > 0:
        print(f"Marcadores STag detectados com IDs: {ids}")
        # Desenha os contornos dos marcadores na imagem e adiciona os IDs
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0) # Cor azul para o texto
        thickness = 2
        for i, corner in enumerate(corners):
            pts = np.array(corner, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(image, [pts], True, (0,255,0), 3)
            centroid = np.mean(pts, axis=0)
            cv2.putText(image, str(ids[i][0]), (int(centroid[0][0]), int(centroid[0][1])), font, font_scale, font_color, thickness)
        cv2.imshow('Marcadores STag Detectados', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhum marcador STag foi encontrado.")

if __name__ == "__main__":
    image_path = input("Por favor, insira o caminho da imagem: ")
    detect_stag_markers(image_path)
