import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem
image = cv2.imread('./Imagens/biel.png', 0)  # Certifique-se de substituir 'sua_imagem.jpg' pelo caminho da sua imagem

# Aplicar a limiarização
threshold_value = 128  # Você pode ajustar esse valor de acordo com suas necessidades
_, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Exibir a imagem original e a imagem limiarizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Imagem Limiarizada')

plt.show()