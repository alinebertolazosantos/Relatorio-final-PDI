import cv2
import numpy as np
import matplotlib.pyplot as plt

#Colocando as imagens 
paths = [
    './imagem/{}.jpeg'.format(i)
    for i in ['b', 'c', 'd', 'e']
]

# Ler as imagens em escala de cinza
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]

# Definindo o elemento estruturante
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Iterando sobre cada imagem
for idx, img in enumerate(images):
    plt.figure(figsize=(15, 4))

    # Abertura
    img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, se_disk)

    # Fechamento
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se_disk)

    # Subplot 1: Imagem original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(chr(98 + idx) + ') Original')
    plt.axis('off')

    # Subplot 2: Imagem após abertura
    plt.subplot(1, 3, 2)
    plt.imshow(img_open, cmap='gray')
    plt.title(chr(98 + idx) + ') Abertura')
    plt.axis('off')

    # Subplot 3: Imagem após fechamento
    plt.subplot(1, 3, 3)
    plt.imshow(img_close, cmap='gray')
    plt.title(chr(98 + idx) + ') Fechamento')
    plt.axis('off')

    plt.tight_layout()
    plt.show()