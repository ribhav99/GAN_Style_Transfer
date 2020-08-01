import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from VAE import VAE
from dataloader import get_data_loader
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(recon_x, x, mu, logsigma):
    BCE = ((recon_x -  x)**2).sum()
    KLD = -0.5*torch.sum(1 + 2*logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

def train_VAE_1_step(model,other_VAE,optim, data):
    optim.zero_grad()
    recon_batch, mu, logsig = model(data)
    fake = other_VAE.decode(model.reparameterize(mu,logsig))
    fake_batch, fake_mu, fake_logsig = other_VAE(fake)
    loss = loss_function(recon_batch, data, mu, logsig) + 0.5*loss_function(fake_batch, fake, fake_mu,fake_logsig)
    loss.backward()
    optim.step()
    return loss.item()

def train(args):
    full_data = get_data_loader(args)
    VAE_human = VAE(512).to(device)
    VAE_cartoon = VAE(512).to(device)
    optimiser_human = optim.Adam(VAE_human.parameters(), lr=0.0002)
    optimiser_cartoon = optim.Adam(VAE_cartoon.parameters(),lr=0.0002)
    VAE_human.train()
    VAE_cartoon.train()
    print("Start Training....")
    for epoch in trange(args.num_epochs):
        total_VAE_human_loss = 0.0
        total_VAE_cartoon_loss = 0.0
        total_data = 0
        for batch_num, data in enumerate(full_data):
            human, cartoon = data[0].to(device), data[1].to(device) # x is cartoon, y is human
            total_data += human.shape[0]
            total_VAE_human_loss += train_VAE_1_step(VAE_human, VAE_cartoon,optimiser_human,human)
            total_VAE_cartoon_loss += train_VAE_1_step(VAE_cartoon, VAE_human,optimiser_cartoon,cartoon)
        avg_VAE_human_loss = total_VAE_human_loss / total_data
        avg_VAE_cartoon_loss = total_VAE_cartoon_loss / total_data
        print("Avg VAE Cartoon Loss: {}".format(avg_VAE_cartoon_loss))
        print("Avg VAE Human Loss: {}".format(avg_VAE_human_loss))
        # torch.save(VAE_human.state_dict(), "vaehuman.pt")
        # torch.save(VAE_cartoon.state_dict(), "vaecartoon.pt")
