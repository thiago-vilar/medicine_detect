import pickle
import matplotlib.pyplot as plt
import numpy as np

def interpret_chain_code(chain_code):
    """ Converte uma lista de códigos de cadeia em coordenadas x, y. """
    moves = {
        0: (1, 1),   # bottom-right
        1: (0, 1),   # right
        2: (-1, 1),  # top-right
        3: (-1, 0),  # left
        4: (-1, -1), # top-left
        5: (0, -1),  # left
        6: (1, -1),  # bottom-left
        7: (1, 0)    # bottom
    }
    x, y = 0, 0  # Começar no centro
    coordinates = [(x, y)]
    for code in chain_code:
        if code in moves:
            dx, dy = moves[code]
            x += dx
            y += dy
            coordinates.append((x, y))
    return np.array(coordinates)

def main():
    filename = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    try:
        with open(filename, 'rb') as f:
            chain_code = pickle.load(f)

        if not isinstance(chain_code, list):
            print("Formato de dados incorreto. Esperado uma lista de códigos numéricos.")
            return

        assinatura = interpret_chain_code(chain_code)

        x, y = assinatura[:, 0], assinatura[:, 1]
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', markersize=5, linestyle='-')  
        plt.title('Assinatura do Contorno')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')
        plt.gca().invert_yaxis()  
        plt.axis('equal')  
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao carregar ou exibir a assinatura: {e}")

if __name__ == "__main__":
    main()
