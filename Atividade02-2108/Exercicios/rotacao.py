import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cam = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    #Inclinação da foto lena
    lena_rotacao = lena.copy()
    lena_np_rotacao_45  = ndimage.rotate(lena_rotacao, -45, cval=128)
    lena_np_rotacao_90 = ndimage.rotate(lena_rotacao, -90, cval=128)
    lena_np_rotacao_100  = ndimage.rotate(lena_rotacao, -100, cval=128)

    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("Lena rotação 45")
    plt2.title.set_text("Lena rotação 90")
    plt3.title.set_text("Lena rotação 100")

    plt1.imshow(lena_np_rotacao_45, cmap="gray")
    plt2.imshow(lena_np_rotacao_90, cmap="gray")
    plt3.imshow(lena_np_rotacao_100, cmap="gray")
    plt.subplots_adjust(wspace=0.5)

    #Inclinação da foto cameraman
    cameraman_rotacao = cam.copy()
    cameraman_np_rotacao_45  = ndimage.rotate(cameraman_rotacao, -45, cval=128)
    cameraman_np_rotacao_90  = ndimage.rotate(cameraman_rotacao, -90, cval=128)
    cameraman_np_rotacao_100  = ndimage.rotate(cameraman_rotacao, -100, cval=128)

    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("Cameraman rotação 45")
    plt2.title.set_text("Cameraman rotação 90")
    plt3.title.set_text("Cameraman rotação 100")

    plt1.imshow(cameraman_np_rotacao_45, cmap="gray")
    plt2.imshow(cameraman_np_rotacao_90, cmap="gray")
    plt3.imshow(cameraman_np_rotacao_100, cmap="gray")
    plt.subplots_adjust(wspace=0.5)

    #Inclinação da foto house
    house_rotacao = house.copy()
    house_np_rotacao_45  = ndimage.rotate(house_rotacao, -45, cval=128)
    house_np_rotacao_90  = ndimage.rotate(house_rotacao, -90, cval=128)
    house_np_rotacao_100  = ndimage.rotate(house_rotacao, -100, cval=128)

    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("House rotação 45")
    plt2.title.set_text("House rotação 90")
    plt3.title.set_text("House rotação 100")
    plt.subplots_adjust(wspace=0.5)

    plt1.imshow(house_np_rotacao_45, cmap="gray")
    plt2.imshow(house_np_rotacao_90, cmap="gray")
    plt3.imshow(house_np_rotacao_100, cmap="gray")

    plt.show()

if __name__ == "__main__":
    main()