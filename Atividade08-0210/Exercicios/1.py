import cv2
import numpy as np
import matplotlib.pyplot as plt

#Colocando as imagens 
paths = [
    './imagem/{}.jpeg'.format(i) 
    for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]

# Ler as imagens em escala de cinza
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]

# Definindo os elementos estruturantes
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
se_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Linha horizontal

# Iterando sobre cada imagem
for idx, img in enumerate(images):
    plt.figure(figsize=(15, 8))

    # Implementando Erosão
    img_eroded_disk = cv2.erode(img, se_disk)
    img_eroded_cross = cv2.erode(img, se_cross)
    img_eroded_line = cv2.erode(img, se_line)

    # Implementando Dilatação
    img_dilated_disk = cv2.dilate(img, se_disk)
    img_dilated_cross = cv2.dilate(img, se_cross)
    img_dilated_line = cv2.dilate(img, se_line)

    #subplots------------------------------------
    # Subplot 1: Imagem original
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(chr(97 + idx) + ') Original')
    plt.axis('off')

    # Subplots 2-4: Imagens erodidas
    for i, eroded in enumerate([img_eroded_disk, img_eroded_cross, img_eroded_line]):
        plt.subplot(3, 4, i+2)
        plt.imshow(eroded, cmap='gray')
        plt.title(chr(97 + idx) + f') Eroded ({["disk", "cross", "line"][i]})')
        plt.axis('off')

    # Subplots 5-7: Imagens dilatadas
    for i, dilated in enumerate([img_dilated_disk, img_dilated_cross, img_dilated_line]):
        plt.subplot(3, 4, i+5)
        plt.imshow(dilated, cmap='gray')
        plt.title(chr(97 + idx) + f') Dilated ({["disk", "cross", "line"][i]})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()