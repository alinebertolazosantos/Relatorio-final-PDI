#1 - e) Aplique uma rotação de 40º no quadrado e repita os passo b-d

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Imagem 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# quadrado branco
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# Aplicando rotação de 40º
rows, cols = imagem.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 40, 1)
imagem_rotacionada = cv2.warpAffine(imagem, M, (cols, rows))

# calculando e vizualizando o espectro de Fourier (amplitudes) da imagem rotacionada
f_rot = np.fft.fft2(imagem_rotacionada)
fshift_rot = np.fft.fftshift(f_rot)
magnitude_spectrum_rot = 20*np.log(np.abs(fshift_rot))

# c)calculando e vizualizando o espectro de Fourier (fases) da imagem rotacionada
fase_spectrum_rot = np.angle(fshift_rot)

# Utilizando subplots para exibir as imagens lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Imagem rotacionada
axs[0].imshow(imagem_rotacionada, cmap='gray')
axs[0].set_title('Imagem Rotacionada')
axs[0].axis('off')

# Espectro de Amplitude
axs[1].imshow(magnitude_spectrum_rot, cmap='gray')
axs[1].set_title('Espectro de Fourier - Amplitude')
axs[1].axis('off')

# Espectro de Fase
axs[2].imshow(fase_spectrum_rot, cmap='gray')
axs[2].set_title('Espectro de Fourier - Fase')
axs[2].axis('off')

plt.tight_layout()
plt.show()
