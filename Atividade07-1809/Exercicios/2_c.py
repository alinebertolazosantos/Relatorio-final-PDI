#2 - c) a imagem de cada filtro

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

def create_filters(image, cutoff):
    # Filtro passa-baixa Ideal
    ideal_filter = ideal_lowpass_filter(image, cutoff)

    # Filtro passa-baixa Butterworth
    butter_filter = butterworth_lowpass_filter(image, cutoff)

    # Filtro passa-baixa Gaussiano
    gaussian_filter = gaussian_lowpass_filter(image, cutoff)

    return ideal_filter, butter_filter, gaussian_filter

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def ideal_lowpass_filter(image, cutoff):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < cutoff:
                filter[y, x] = 1
    return filter

def butterworth_lowpass_filter(image, cutoff, order=2):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    
    for x in range(cols):
        for y in range(rows):
            filter[y, x] = 1 / (1 + (distance((y, x), center) / cutoff) ** (2 * order))
    return filter

def gaussian_lowpass_filter(image, cutoff):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    
    for x in range(cols):
        for y in range(rows):
            filter[y, x] = np.exp(-(distance((y, x), center) ** 2) / (2 * (cutoff ** 2)))
    return filter
cutoff = 30  

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

cutoff = 100  # Ajuste conforme necessário

images = [sinc_original, sinc_rot, sinc_rot2, sinc_trans]
titles = ['Original', 'Rotacionada (40º)', 'Rotacionada (20º)', 'Transladada']

for idx, image in enumerate(images):
    # Cria os filtros
    ideal_filter, butter_filter, gaussian_filter = create_filters(image, cutoff)
    
    # Normaliza para melhor visualização
    ideal_filter = normalize_image(ideal_filter)
    butter_filter = normalize_image(butter_filter)
    gaussian_filter = normalize_image(gaussian_filter)
    
    # Exibe os filtros
    plt.figure(figsize=(15, 5))
    
    # Filtro Ideal
    plt.subplot(131)
    plt.imshow(ideal_filter, cmap='gray')
    plt.title(f'Filtro Ideal - {titles[idx]}')
    plt.axis('off')

    # Filtro Butterworth
    plt.subplot(132)
    plt.imshow(butter_filter, cmap='gray')
    plt.title(f'Filtro Butterworth - {titles[idx]}')
    plt.axis('off')

    # Filtro Gaussiano
    plt.subplot(133)
    plt.imshow(gaussian_filter, cmap='gray')
    plt.title(f'Filtro Gaussiano - {titles[idx]}')
    plt.axis('off')

    # Ajusta o layout e mostra a figura
    plt.tight_layout()
    plt.show()

