import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from . import mnist_dataloader

__add__  = ['view_data']

images, labels = next(iter(mnist_dataloader(batch_size=16)[0]))

def image_display(image, title=None):
    image = image/2 + 0.5
    numpy_image = image.numpy()
    transposed_numpy_image = np.transpose(numpy_image, (1, 2, 0))
    plt.figure(figsize=(30, 10))
    plt.imshow(transposed_numpy_image)
    plt.yticks([])
    plt.xticks([])
    if title:
        plt.title(title)
    plt.show
    
def view_data():
    return image_display(torchvision.utils.make_grid(images))
