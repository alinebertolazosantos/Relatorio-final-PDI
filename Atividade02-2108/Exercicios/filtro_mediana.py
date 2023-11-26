import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage


#Função de exibição de comparação de imagens
def print_image(original, rect):
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Mediana Aplicada")
    plt.imshow(rect, cmap="gray")
    plt.show()

#Função mediana
def median(array, array_nd):
 # operação por vizinhança
    l = array.shape[0]
    c = array.shape[1]
    k = 3
    #print("Imagem")

    #print(array[0:5,0:5])
    for x in range(k, l-k): #linhas
        for y in range(k, c-k): #colunas
            s_xy = array[x-k:x+k+1,y-k:y+k+1]
            array_nd[x,y] = np.median(s_xy).astype(int)
            #print('janela')
            #print(s_xy)
    return array_nd

            
#Função principal 
def main():
    path = "C:/Users/Berto/Documents/Facul/Processamento_Digital_Imagem/image"

    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cameraman = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    # Conversão para array
    lena_array = np.array(lena)
    lena_nd = np.zeros(lena_array.shape)
    house_array = np.array(house)
    house_nd = np.zeros(house_array.shape)
    cameraman_array = np.array(cameraman)
    cameraman_nd = np.zeros(cameraman_array.shape)

    #Chamada da função media 
    lena_media = median(lena_array.copy(), lena_nd.copy())
    cameraman_media = median(cameraman_array.copy(), cameraman_nd.copy())
    house_media = median(house_array.copy(), house_nd.copy())

    # Exibição com matplotlib 
    # Lena
    plt.figure("Lena")
    print_image(lena_array, lena_media)

    # house
    plt.figure("house")
    print_image(house_array, house_media)

    # Cameraman
    plt.figure("Cameraman")
    print_image(cameraman_array, cameraman_media)

if __name__ == "__main__":
    main()