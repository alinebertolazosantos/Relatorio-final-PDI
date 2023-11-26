import numpy as np 
import matplotlib.pyplot as plt

coluna = 25
linhas = 14 
imagemMatriz = np.zeros([linhas, coluna])
print (imagemMatriz.shape)

#fazendo letra 1
imagemMatriz[9:14,0]=255
imagemMatriz[5:9,1]=255
imagemMatriz[2:5,2]=255
imagemMatriz[2:5,3]=255
imagemMatriz[2:5,4]=255
imagemMatriz[5:9,5]=255
imagemMatriz[9:14,6]=255
imagemMatriz[9,0:6]=255

#Fazendo a letra B
imagemMatriz[2:14,8]=255
imagemMatriz[2,9:14]=255
imagemMatriz[3:8,14]=255
imagemMatriz[8,9:14]=255
imagemMatriz[9,14:15]=255
imagemMatriz[9:13,14]=255
imagemMatriz[13,8:14]=255

#Fazendo a letra S
imagemMatriz[2, 17:24]=255
imagemMatriz[7, 17:24]=255
imagemMatriz[2:7, 17]=255
imagemMatriz[13,17:24]=255
imagemMatriz[8:13, 23]=255

plt.imshow(imagemMatriz, cmap='gray')
plt.show()


