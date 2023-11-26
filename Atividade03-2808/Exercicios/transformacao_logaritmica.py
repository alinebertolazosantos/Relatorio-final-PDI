import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Mandando a imagem para a variavel 
enhance_path = 'C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image/enhanceme.gif'
fractured_path = 'C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image/fractured.tif'
img = cv2.imread(enhance_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(fractured_path,cv2.IMREAD_GRAYSCALE)

#Abrindo imagem
image_pil = Image.open(enhance_path)
image_pil2 = Image.open(fractured_path)


# Convertendo a imagem PIL para um array numpy
img = np.array(image_pil)
img2 = np.array(image_pil2)

def log_transform(c, img):
 return c * np.log(1 + img)
c_values = [1, 5, 10, 20]

def log_transformF(c, img2):
 return c * np.log(1 + img2)
c_valuesF = [1, 5, 10, 20]

for c in c_values:
 transformed_img = log_transform(c, img)
 plt.imshow(transformed_img, cmap='gray')
 plt.title(f'c = {c}')
 plt.show()

for c in c_valuesF:
 transformed_imgF = log_transform(c, img2)
 plt.imshow(transformed_imgF, cmap='gray')
 plt.title(f'c = {c}')
 plt.show()