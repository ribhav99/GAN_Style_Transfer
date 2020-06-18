import numpy as np 
import PIL.Image
from PIL.ImageOps import grayscale
from skimage.transform import resize
import os

human_faces_data = []
human_faces_target = []
cartoon_data = []

human_faces_data_directory = ""
cartoon_data_directory = ""
download_directory = ""

image_dimensions = (256, 256)
cartoon_dimensions = (32, 32)

def load_human_faces_data(directory):
    for file in os.scandir(directory):
        try:
            image = PIL.Image.open(file) # or data_directory + "/" + file.  ALSO, convert to numpy array or not?
            image = resize(image, ouputsize=image_dimensions)
            human_faces_data.append(image)
            human_faces_target.append(1)
        except:
            pass

def load_cartoon_data(directory):
    for file in os.scandir(directory):
        try:
            image = PIL.Image.open(file) # or data_directory + "/" + file.  ALSO, convert to numpy array or not?
            image = resize(image, ouputsize=cartoon_dimensions)
            cartoon_data.append(image)
        except:
            pass

def gather_cartoon_data(directory):
    pass