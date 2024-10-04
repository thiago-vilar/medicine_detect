import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    filename = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    try:
        # Carregar a assinatura do contorno do arquivo .pkl
        with open(filename, 'rb') as f:
            assinatura = pickle.load(f)

        # Assegurar que a assinatura esteja em formato de array numpy
        if not isinstance(assinatura, np.ndarray):
            assinatura = np.array(assinatura)

        # Verificar se o shape é adequado (N, 2)
        if assinatura.ndim != 2 or assinatura.shape[1] != 2:
            print("Dados inesperados. Esperado um array de coordenadas com shape (N, 2).")
            return
        
        # Separar as coordenadas x e y
        x, y = assinatura[:, 0], assinatura[:, 1]

        # Plotar o contorno usando scatter para garantir que os pontos não se conectem
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='blue', s=5)  # Usar scatter em vez de plot com '-o' para evitar conectar pontos
        plt.title('Assinatura do Contorno')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')
        plt.gca().invert_yaxis()  # Inverter o eixo Y para alinhamento correto da visualização
        plt.axis('equal')  # Manter a proporção dos eixos
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao carregar ou exibir a assinatura: {e}")

if __name__ == "__main__":
    main()
