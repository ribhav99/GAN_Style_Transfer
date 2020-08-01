import torch
import torch.nn as nn
from init_helper import ResidualBlock, get_up_conv_block , get_down_conv_block
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, latent_dim = 512):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        channel_list = [3,32,64,128,256]
        act_fn = "relu"
        downconv = []
        downconv += get_down_conv_block(3,32,4,act_fn,"none", False) 
        downconv += get_down_conv_block(32,64,4,act_fn,"none", False) 
        downconv += get_down_conv_block(64,128,4,act_fn,"none", False) 
        downconv += get_down_conv_block(128,256,4,act_fn,"none", False) 
        for i in range(3):
            downconv += [ResidualBlock(256,"none")]
        downconv += [nn.Flatten(), nn.Linear(16384, 2*latent_dim)]
        self.encoder = nn.Sequential(*downconv)
        self.reshape = nn.Sequential(nn.Linear(latent_dim, 16384), nn.ReLU(True))
        upconv = []
        upconv += get_up_conv_block(256,128,4,act_fn,"none", False) 
        upconv += get_up_conv_block(128,64,4,act_fn,"none", False) 
        upconv += get_up_conv_block(64,32,4,act_fn,"none", False) 
        upconv += get_up_conv_block(32,3,4,"tanh","none", False) 
        self.decoder = nn.Sequential(*upconv) 

    def encode(self,x):
        x = self.encoder(x)
        return x[...,:self.latent_dim], x[...,self.latent_dim:]

    def decode(self,x):
        x = self.reshape(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 256,8,8)
        x = self.decoder(x)
        return x

    def reparameterize(self,mu, logsig):
        std = torch.exp(logsig)
        eps = torch.randn_like(mu)
        return mu + (eps * std)

    def forward(self,x):
        mu, logsig = self.encode(x)
        z = self.reparameterize(mu, logsig)
        output = self.decode(z)
        return output, mu, logsig

def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = ((recon_x -  x)**2).sum()
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

def trainVAE(num_epoch, data):
    model = VAE().to(device)
    optim = Adam(model.parameters(), lr = 1e-3)
    model.train()
    for i in range(num_epoch):
        train_loss = 0.0
        optim.zero_grad()
        recon_batch, mu, logsig = model(data)
        loss = loss_function(recon_batch, data, mu, logsig)
        loss.backward()
        train_loss += loss.item()
        optim.step()
        print(train_loss)
    return model
