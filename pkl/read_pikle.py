import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    filename = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    try:

        with open(filename, 'rb') as f:
            assinatura = pickle.load(f)


        if not isinstance(assinatura, np.ndarray):
            assinatura = np.array(assinatura)

      
        if assinatura.ndim != 2 or assinatura.shape[1] != 2:
            print("Dados inesperados. Esperado um array de coordenadas com shape (N, 2).")
            return

        x, y = assinatura[:, 0], assinatura[:, 1]

 
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='blue', s=5)  
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
