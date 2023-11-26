import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def main():
    #variaveis recebem o endereço das imagens
    lena = np.array(Image.open('../image/lena_gray_512.tif'))
    cam = np.array(Image.open('../image/cameraman.tif'))
    house = np.array(Image.open('../image/casa.tif'))

    #Mudança de zoom lena 1,5 e 2,5 lena_change_sizes
    lena_change_sizes = lena.copy()
    lena_zoom = ndimage.zoom(lena_change_sizes, (2.5, 2.5))
    lena_shrink = ndimage.zoom(lena_change_sizes, (1/1.5, 1/1.5))
    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("Lena Original")
    plt2.title.set_text("Lena com Zoom 2.5")
    plt3.title.set_text("Lena com Shirk 1.5")
    plt.subplots_adjust(wspace=0.5)

    plt1.imshow(lena, cmap="gray")
    plt2.imshow(lena_zoom, cmap="gray")
    plt3.imshow(lena_shrink, cmap="gray")
    
    cameraman_change_size = cam.copy()
    cameraman_zoom = ndimage.zoom(cameraman_change_size, (2.5, 2.5))
    cameraman_shrink = ndimage.zoom(cameraman_change_size, (1/1.5, 1/1.5))
    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("Cameraman com Original")
    plt2.title.set_text("Cameraman com Zoom 2.5")
    plt3.title.set_text("Cameraman com Shirk 1.5")
    plt.subplots_adjust(wspace=0.5)

    plt1.imshow(cam, cmap="gray")
    plt2.imshow(cameraman_zoom, cmap="gray")
    plt3.imshow(cameraman_shrink, cmap="gray")

    house_chance_sizes = house.copy()
    house_zoom = ndimage.zoom(house_chance_sizes, (2.5, 2.5))
    house_shrink = ndimage.zoom(house_chance_sizes, (1/1.5, 1/1.5))
    fig = plt.figure(figsize=(10, 5))
    plt1 = plt.subplot(1, 3, 1)
    plt2 = plt.subplot(1, 3, 2)
    plt3 = plt.subplot(1, 3, 3)
    plt1.title.set_text("House Original")
    plt2.title.set_text("House com Zoom 2.5")
    plt3.title.set_text("House com Shirk 1.5")
    plt.subplots_adjust(wspace=0.5)

    plt1.imshow(house, cmap="gray")
    plt2.imshow(house_zoom, cmap="gray")
    plt3.imshow(house_shrink, cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()