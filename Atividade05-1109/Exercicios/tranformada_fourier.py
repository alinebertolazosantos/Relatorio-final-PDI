import numpy as np
import matplotlib.pyplot as plt
import cv2

# Carreguando imagens
#img = cv2.imread("./Imagem/Quadrado.jpeg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("./Imagem/Jordam.jpeg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("./Imagem/Olho.jpeg", cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("./Imagem/Lena.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("./Imagem/Car.jpeg", cv2.IMREAD_GRAYSCALE)



# Função para calcular a Transformada de Fourier 2D e plotar o espectro e a fase
def calcular_e_plotar_fft(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(transformada_fourier)

    # Calcule a fase
    fase = np.angle(transformada_fourier)

    # Plote o espectro de magnitude e a fase
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log(1 + espectro_magnitude), cmap='gray')
    plt.title('Espectro da Magnitude')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(fase, cmap='gray')
    plt.title('Fase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Chamada da função para cada imagem
calcular_e_plotar_fft(img, "Imagem Lena")