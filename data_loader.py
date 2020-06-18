import numpy as np 
import PIL.Image
from PIL.ImageOps import grayscale
from skimage.transform import resize
import os

data = []
target = []
data_directory = ""

image_dimensions = (256, 256)
cartoon_dimensions = (32, 32)

def load_data(directory):
    for file in os.scandir(directory):
        try:
            image = PIL.Image.open(file) # or data_directory + "/" + file.  ALSO, convert to numpy array or not?
            image = resize(image, ouputsize=image_dimensions)
            data.append(image)
            target.append(1)
