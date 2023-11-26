import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('./Imagens/biel.png', 0)  # Substitua 'sua_imagem.jpg' pelo caminho da sua imagem

# 2.1. Aplicar filtro de borramento gaussiano
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # O segundo argumento é o tamanho do kernel gaussiano

# Aplicar o detector de bordas Canny na imagem original
canny_edges = cv2.Canny(image, 100, 200)  # Você pode ajustar os limiares conforme necessário

# Aplicar o detector de bordas Canny na imagem borradad
canny_edges_blurred = cv2.Canny(blurred_image, 100, 200)  # Novamente, ajuste os limiares conforme necessário

# Exibir as imagens
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(2, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Imagem Borradad (Gaussiano)')

plt.subplot(2, 2, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Detecção de Bordas (Canny) na Imagem Original')

plt.subplot(2, 2, 4)
plt.imshow(canny_edges_blurred, cmap='gray')
plt.title('Detecção de Bordas (Canny) na Imagem Borradad')

plt.show()
