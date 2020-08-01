import numpy as np
import skimage.io as io
import torch
import os
import plot_features
from VAE import VAE
from Generator import Generator

def load_img(args,pathimg):
    image = io.imread(pathimg).astype("float32")
    if args.gray:
        image = rgb2gray(image)
    image = (image / 127.5) - 1
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    elif len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    return image

def _load_helper(args,root_path):
    images = []
    names = []
    for root, dirs, files in os.walk(root_path):
       for name in files:
           if name.endswith(".png") or name.endswith(".jpg"):
               images.append(load_img(args,os.path.join(root_path, name)))
               names.append(name)
    return torch.tensor(images, dtype=torch.float32), names

def load_x_y(args,x_dir, y_dir):
    x, x_names = _load_helper(args,x_dir)
    y , y_names = _load_helper(args,y_dir)
    return x,x_names, y, y_names

def gen_images(root_dir, x, y, x_names, y_names, args,save = True):
    g_x_y = Generator(args)
    g_x_y.load_state_dict(torch.load(os.path.join(root_dir,"g_x_y.pt")))
    with torch.no_grad():
        generated_y = g_x_y(x).numpy()
        feature_node_x_y = g_x_y.go[0]
    if save:
        save_images(os.path.join(root_dir, "generated_y"), generated_y, y_names)
    return feature_node_x_y

def gen_images_VAE(root_dir, x, y, x_names, y_names, args, save = True):
    def VAE_generate(model, othervae, x):
        sample_mu, sample_logsig = model.encode(x)
        matrix_of_img = othervae.decode(model.reparameterize(sample_mu,sample_logsig))
        return matrix_of_img.detach().numpy()
    VAE_human = VAE()
    VAE_cartoon = VAE()
    VAE_human.load_state_dict(torch.load(os.path.join(root_dir,"vaehuman.pt")))
    VAE_cartoon.load_state_dict(torch.load(os.path.join(root_dir,"vaecartoon.pt")))
    with torch.no_grad():
        generated_y = VAE_generate(VAE_cartoon, VAE_human, x)
        feature_node_x_y = VAE_cartoon.encoder[0]
    if save:
        save_images(os.path.join(root_dir,"generated_y"), generated_y, y_names)
    return feature_node_x_y

def save_images(foldername, images, names):
    dir_path = foldername + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i in range(images.shape[0]):
        image = np.moveaxis(images[i],0,-1)
        io.imsave(dir_path + names[i] + ".png",image)

def extract_feature(root_dir, image, kernel):
    output = kernel(image)
    output = output.squeeze(0).detach().numpy()
    for i in range(output.shape[0]):
        io.imsave(os.path.join(root_dir, "feature_{}.png".format(i)), output[i])

def evaluate(root_dir, args, isVAE = False):
    x_dir = "../testCartoon/"
    y_dir = "../testHuman/"
    x, x_names, y, y_names = load_x_y(args,x_dir, y_dir)
    if isVAE:
        feature_node_x_y = gen_images_VAE(root_dir,x,y,x_names, y_names, args, True)
        desired = x_names.index("4.png")
    else:
        feature_node_x_y = gen_images(root_dir,x,y,x_names, y_names, args, True)
        desired = x_names.index("20.png")
    to_extract_feat = x[desired].unsqueeze(0)
    extract_feature(root_dir, to_extract_feat, feature_node_x_y)
    plot_features.plot_features(root_dir)
