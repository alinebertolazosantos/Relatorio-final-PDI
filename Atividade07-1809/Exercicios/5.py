#5 Efetue o mesmo que se pede no item 4, mas use o filtro passa-alta em vez do filtro passa-baixa.

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

images = [sinc_original, sinc_rot, sinc_rot2, sinc_trans]
titles = ['Original', 'Rotacionada (40º)', 'Rotacionada (20º)', 'Transladada']

def fourier_spectrum(image):
 # Computa a transformada de Fourier 2D
 f = np.fft.fft2(image)
 # Centraliza as frequências baixas
 fshift = np.fft.fftshift(f)
 # Calcula a magnitude e aplica o logaritmo para melhor visualização
 magnitude_spectrum = np.log(np.abs(fshift) + 1)
 return magnitude_spectrum

def ideal_highpass_filter(image, cutoff):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2

    filter = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= cutoff:
                filter[i, j] = 0
    return filter

def butterworth_highpass_filter(image, cutoff, order=2):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2

    filter = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            filter[i, j] = 1 / (1 + (cutoff / distance) ** (2 * order))

    return filter

def gaussian_highpass_filter(image, cutoff):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2

    filter = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            filter[i, j] -= np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))

    return filter

def apply_filter(image, filter):
    # Aqui assumo que você está usando a Transformada de Fourier para aplicar o filtro.
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalizando a imagem resultante para o intervalo [0, 255]
    img_normalized = np.divide(img_back - np.min(img_back), np.max(img_back) - np.min(img_back)) * 255
    
    return img_normalized

cutoffs = [0.01, 0.05, 0.5]

for cutoff in cutoffs:
    for idx, image in enumerate(images):
        # Fourier
        spectrum = fourier_spectrum(image)

        # Criação dos filtros
        ideal_hp = ideal_highpass_filter(image, cutoff)
        butter_hp = butterworth_highpass_filter(image, cutoff)
        gaussian_hp = gaussian_highpass_filter(image, cutoff)

        # Aplicação dos filtros
        result_ideal = apply_filter(image, ideal_hp)
        result_butter = apply_filter(image, butter_hp)
        result_gaussian = apply_filter(image, gaussian_hp)

        # Exibição
        plt.figure(figsize=(20, 10))

        # Imagem original
        plt.subplot(4, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'a) Imagem {titles[idx]}')
        plt.axis('off')

        # Espectro de Fourier
        plt.subplot(4, 4, 2)
        plt.imshow(spectrum, cmap='gray')
        plt.title('b) Espectro de Fourier')
        plt.axis('off')

        # Filtro passa-alta Ideal
        plt.subplot(4, 4, 3)
        plt.imshow(ideal_hp, cmap='gray')
        plt.title(f'c) Filtro Ideal Passa-Alta (D0={cutoff})')
        plt.axis('off')

        # Resultado filtro Ideal
        plt.subplot(4, 4, 4)
        plt.imshow(result_ideal, cmap='gray')
        plt.title('d) Após Filtro Ideal')
        plt.axis('off')

        # Filtro passa-alta Butterworth
        plt.subplot(4, 4, 7)
        plt.imshow(butter_hp, cmap='gray')
        plt.title(f'c) Filtro Butterworth Passa-Alta (D0={cutoff})')
        plt.axis('off')

        # Resultado filtro Butterworth
        plt.subplot(4, 4, 8)
        plt.imshow(result_butter, cmap='gray')
        plt.title('d) Após Filtro Butterworth')
        plt.axis('off')

        # Filtro passa-alta Gaussiano
        plt.subplot(4, 4, 11)
        plt.imshow(gaussian_hp, cmap='gray')
        plt.title(f'c) Filtro Gaussiano Passa-Alta (D0={cutoff})')
        plt.axis('off')

        # Resultado filtro Gaussiano
        plt.subplot(4, 4, 12)
        plt.imshow(result_gaussian, cmap='gray')
        plt.title('d) Após Filtro Gaussiano')
        plt.axis('off')

        # Ajusta o layout e mostra a figura
        plt.tight_layout()
        plt.show()
