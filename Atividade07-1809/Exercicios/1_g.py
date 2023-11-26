#1 - g) Aplique um zoom na imagem e repita os passo b-d;

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Imagem 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# quadrado branco
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# Aplicando zoom: reduzindo a imagem em 50% e depois aumentando para o tamanho o
imagem_zoom = cv2.resize(imagem, (512, 512))
imagem_zoom = cv2.resize(imagem_zoom, (1024, 1024))

# Calculando o espectro de Fourier (amplitudes) da imagem com zoom
f_zoom =  np.fft.fft2(imagem_zoom)
fshift_zoom = np.fft.fftshift(f_zoom)

magnitude_spectrum_zoom = 20*np.log(np.abs(fshift_zoom) + 1) 

#  Calculando o espectro de Fourier (fases) da imagem com zoom
fase_spectrum_zoom = np.angle(fshift_zoom)

# Usando subplots para exibir as imagens lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Imagem com zoom
axs[0].imshow(imagem_zoom, cmap='gray')
axs[0].set_title('Imagem com Zoom')
axs[0].axis('off')

# Espectro de Amplitude
axs[1].imshow(magnitude_spectrum_zoom, cmap='gray')
axs[1].set_title('Espectro de Fourier - Amplitude')
axs[1].axis('off')

# Espectro de Fase
axs[2].imshow(fase_spectrum_zoom, cmap='gray')
axs[2].set_title('Espectro de Fourier - Fase')
axs[2].axis('off')
plt.tight_layout()
plt.show()