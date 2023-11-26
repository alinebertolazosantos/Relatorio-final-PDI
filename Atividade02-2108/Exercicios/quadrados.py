import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cam = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    #Inserindo os quadrados nos cantos na imagem da lena 
    sizeLena = lena.shape[0]
    lenaCQuadrado = lena.copy()
    lenaCQuadrado[0:10, 0:10] = 255
    lenaCQuadrado[0:10, sizeLena-10:sizeLena] = 255 
    lenaCQuadrado[sizeLena-10:sizeLena, 0:10] = 255
    lenaCQuadrado[sizeLena-10:sizeLena,sizeLena-10:sizeLena] = 255
    plt.imshow(lenaCQuadrado, cmap="gray")
    plt.show()

    #Inserindo os quadrados nos cantos na imagem do cameraman
    sizeCam = cam.shape[0]
    CamCQuadrado = cam.copy()
    CamCQuadrado[0:10, 0:10] = CamCQuadrado[0:10, sizeCam-10:sizeCam] = 255 
    CamCQuadrado[sizeCam-10:sizeCam, 0:10] = CamCQuadrado[sizeCam-10:sizeCam,sizeCam-10:sizeCam] = 255
    plt.imshow(CamCQuadrado, cmap="gray")
    plt.show()

    #Inserindo os quadrados nos cantos na imagem da house
    sizeHouse = house.shape[0]
    HouseCQuadrado = house.copy()
    HouseCQuadrado[0:10, 0:10] = HouseCQuadrado[0:10, sizeHouse-10:sizeHouse] = 255
    HouseCQuadrado[sizeHouse-10:sizeHouse, 0:10] = HouseCQuadrado[sizeHouse-10:sizeHouse,sizeHouse-10:sizeHouse] = 255
    plt.imshow(HouseCQuadrado, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()