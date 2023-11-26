#2 - b) a imagem do spectro de fourier

import cv2
import numpy as np
import matplotlib.pyplot as plt

sinc_original_path = './imagem/sinc_original.png'
sinc_original_menor_path = './imagem/sinc_original_menor.png'
sinc_rot_path = './imagem/sinc_rot.png'
sinc_rot2_path = './imagem/sinc_rot2.png'
sinc_trans_path = './imagem/sinc_trans.png'

# Ler as imagens
sinc_original = cv2.imread(sinc_original_path, cv2.IMREAD_GRAYSCALE)
sinc_original_menor = cv2.imread(sinc_original_menor_path, cv2.IMREAD_GRAYSCALE)
sinc_rot = cv2.imread(sinc_rot_path, cv2.IMREAD_GRAYSCALE)
sinc_rot2 = cv2.imread(sinc_rot2_path, cv2.IMREAD_GRAYSCALE)
sinc_trans = cv2.imread(sinc_trans_path, cv2.IMREAD_GRAYSCALE)

def fourier_spectrum(image):
    # Computa a transformada de Fourier 2D
    f = np.fft.fft2(image)
    # Centraliza as frequências baixas
    fshift = np.fft.fftshift(f)
    # Calcula a magnitude e aplica o logaritmo para melhor visualização
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

# Computa o espectro de Fourier para todas as imagens
spectrum_original = fourier_spectrum(sinc_original)
spectrum_rot = fourier_spectrum(sinc_rot)
spectrum_rot2 = fourier_spectrum(sinc_rot2)
spectrum_trans = fourier_spectrum(sinc_trans)

# Organização das subplots em uma única linha com 4 colunas
plt.figure(figsize=(20, 5)) 

# Espectro de Fourier da Imagem Original
plt.subplot(141)
plt.imshow(spectrum_original, cmap='gray')
plt.title('Espectro de Fourier Original')
plt.axis('off')

# Espectro de Fourier da Imagem Rotacionada 40º
plt.subplot(142)
plt.imshow(spectrum_rot, cmap='gray')
plt.title('Espectro de Fourier Rotacionada (40º)')
plt.axis('off')

# Espectro de Fourier da Imagem Rotacionada 20º
plt.subplot(143)
plt.imshow(spectrum_rot2, cmap='gray')
plt.title('Espectro de Fourier Rotacionada (20º)')
plt.axis('off')

# Espectro de Fourier da Imagem Transladada
plt.subplot(144)
plt.imshow(spectrum_trans, cmap='gray')
plt.title('Espectro de Fourier Transladada')
plt.axis('off')

# Ajusta o layout e mostra a figura
plt.tight_layout()
plt.show()