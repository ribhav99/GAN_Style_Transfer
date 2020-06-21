import numpy as np 
import PIL.Image
from PIL.ImageOps import grayscale
from skimage.transform import resize
import os
from tqdm import tqdm

human_faces_data = []
human_faces_target = []
cartoon_data = []

human_faces_data_directory = ""
cartoon_data_directory = "/Users/RibhavKapur/Downloads/cartoonset10k"

image_dimensions = (256, 256)
cartoon_dimensions = (128, 128)

def load_human_faces_data(directory):
    for file in tqdm(os.listdir(directory)):
        try:
            if file.endswith(".png") or file.endswith(".jpg"):
                image = np.asarray(PIL.Image.open(human_faces_data_directory + "/" + file))
                image = resize(image, output_shape=image_dimensions)
                human_faces_data.append(image)
                human_faces_target.append(1)
        except:
            pass

def load_cartoon_data(directory):
    """
    This works for data downloaded from: https://google.github.io/cartoonset/
    Adds all the images, and resizes them to desired size, to an array
    """
    for file in tqdm(os.listdir(directory)):
        if file.endswith(".png") or file.endswith(".jpg"):
            image = np.asarray(PIL.Image.open(cartoon_data_directory + "/" + file))
            image = resize(image, output_shape=cartoon_dimensions)
            cartoon_data.append(image)



#TESTING DELETE LATER XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
load_cartoon_data(cartoon_data_directory)
print(len(cartoon_data))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX