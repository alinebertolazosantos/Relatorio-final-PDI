import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('./Imagens/biel.png', 0)  # Substitua 'sua_imagem.jpg' pelo caminho da sua imagem

# 2.1. Aplicar filtro de borramento gaussiano
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # O segundo argumento é o tamanho do kernel gaussiano

# Definir diferentes valores de T1 e T2
T1_values = [50, 100, 150]
T2_values = [100, 150, 200]

plt.figure(figsize=(12, 12))
plt.suptitle('Efeito dos parâmetros T1 e T2 na Detecção de Bordas', fontsize=16)

for i, (T1, T2) in enumerate(zip(T1_values, T2_values)):
    # Aplicar o detector de bordas Canny com os valores de T1 e T2
    canny_edges = cv2.Canny(image, T1, T2)

    plt.subplot(3, 3, i + 1)
    plt.imshow(canny_edges, cmap='gray')
    plt.title(f'T1={T1}, T2={T2}')

    plt.subplot(3, 3, i + 4)
    canny_edges_blurred = cv2.Canny(blurred_image, T1, T2)
    plt.imshow(canny_edges_blurred, cmap='gray')
    plt.title(f'T1={T1}, T2={T2} (Borradad)')

plt.show()
