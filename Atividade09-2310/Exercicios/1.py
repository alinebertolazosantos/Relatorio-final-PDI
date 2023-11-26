import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('./Imagens/ponto.jpg', 0)  # Substitua 'sua_imagem.jpg' pelo caminho da sua imagem

# Parâmetros do detector de pontos Harris
block_size = 2  # Tamanho da vizinhança considerada para cada ponto
ksize = 3  # Tamanho do kernel Sobel usado para calcular gradientes
k = 0.04  # Parâmetro de sensibilidade

# Aplicar o detector de pontos Harris
harris_corners = cv2.cornerHarris(image, block_size, ksize, k)

# Normalizar os resultados para visualização
harris_corners = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Definir um limite para destacar os cantos
threshold = 100

# Desenhar círculos nos cantos detectados
image_with_corners = image.copy()
image_with_corners[harris_corners > threshold] = 255

# Exibir a imagem original com os cantos destacados
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(image_with_corners, cmap='gray')
plt.title('Cantos Detectados')

plt.show()
