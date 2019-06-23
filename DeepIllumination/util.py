import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
import cv2

# for debugging
from PIL import Image

def load_image(filepath):
    image = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    if image is None:
        print(filepath)
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
    #if im_show_debug< 5:
    #    im_temp = Image.fromarray(image, 'RGB')
    #    im_temp.show()
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    min = float(image.min())
    max = float(image.max())
    #print("min", min, " ", "max", max)
    image = torch.FloatTensor(image.size()).copy_(image)
    if max != min:
        image.mul_(1.0 / (max - min)) #add_(-min)
    #image = image.mul_(2.0).add_(-1.0)
    #if im_show_debug< 5:
    #    im_temp = Image.fromarray(image.numpy(), 'RGB')
    #    im_temp.show()
    return image


def save_image(image, filename):
    #image = image.add_(1.0).div_(2.0)
    image = image.numpy()
    #image *= 255.0
    image = image.clip(0, 255)
    image = np.transpose(image, (1, 2, 0))
    #image = image.astype(np.uint8)
    imsave(filename, image)
    print ("Image saved as {}".format(filename))

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg",".ppm"])
