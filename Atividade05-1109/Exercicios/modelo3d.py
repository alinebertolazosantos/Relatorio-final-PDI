import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular e plotar o espectro 3D
def plotar_espectro_3D(imagem, titulo):
    
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro 2D
    espectro_2D = np.fft.fftshift(transformada_fourier)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(espectro_2D)

    # Crie uma grade de coordenadas para o espectro 3D
    x = np.fft.fftshift(np.fft.fftfreq(imagem.shape[1]))
    y = np.fft.fftshift(np.fft.fftfreq(imagem.shape[0]))
    X, Y = np.meshgrid(x, y)

    # Plote o espectro 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Espectro 3D - ' + titulo)
    ax.plot_surface(X, Y, np.log(1 + espectro_magnitude), cmap='viridis')

    plt.show()

# Carreguegando a imagem
imgCar = cv2.imread("./Imagem/Car.jpeg", cv2.IMREAD_GRAYSCALE)
imgLen = cv2.imread("./Imagem/lena.jpeg", cv2.IMREAD_GRAYSCALE)
imgNS = cv2.imread("./Imagem/Olho.jpeg", cv2.IMREAD_GRAYSCALE)
imgPeriodic = cv2.imread("./Imagem/Quadrado.jpeg", cv2.IMREAD_GRAYSCALE)
imgSinc = cv2.imread("./Imagem/Jordam.jpeg", cv2.IMREAD_GRAYSCALE)

# Chamada da função para cada imagem
plotar_espectro_3D(imgCar, "3d Carro")
plotar_espectro_3D(imgLen, "3d Lena")
plotar_espectro_3D(imgNS, "3d Olho")
plotar_espectro_3D(imgPeriodic, "3d Quadrado")
plotar_espectro_3D(imgSinc, "3d Homem")