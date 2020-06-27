import torch
import torch.nn as nn
import torch.optim as optim
from Generator import Generator
from Discriminator import Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args, wandb = None):
    full_data = get_data_loader(args, train=True)

    d_x = Discriminator(args).to(device) # x is cartoon, y is human faces
    d_y = Discriminator(args).to(device)
    g_x_y = Generator(args).to(device) #cartoon -> human faces
    g_y_x = Generator(args).to(device) #human faces -> cartoon
    optimiser_d_x = optim.Adam(d_x.parameters(), lr=args.dis_learning_rate)
    optimiser_d_y = optim.Adam(d_y.parameters(), lr=args.dis_learning_rate)
    optimiser_g_x_y = optim.Adam(g_x_y.parameters(), lr=args.gen_learning_rate)
    optimiser_g_y_x = optim.Adam(g_y_x.parameters(), lr=args.gen_learning_rate)
    if args.use_wandb:
        wandb.watch(d_x, log='all')
        wandb.watch(d_y, log='all')
        wandb.watch(g_x_y, log = 'all')
        wandb.watch(g_y_x, log = 'all')
    def cycle_loss(real, reconstructed):
        loss = (torch.abs(real - reconstructed)).mean()
        return args.lambda_cycle * loss 
    d_x.train()
    d_y.train()
    g_x_y.train()
    g_y_x.train()
    print("Start Training....")
    for epoch in trange(args.num_epochs):
        total_d_loss = 0.0
        total_g_x_y_loss = 0.0
        total_g_y_x_loss = 0.0
        total_data = 0
        for batch_num, data in enumerate(full_data):
            y , x = data[0].to(device), data[1].to(device) # x is cartoon, y is human
            total_data += x.shape[0]
            fake_x = g_y_x(y)
            fake_y = g_x_y(x)
            optimiser_d_x.zero_grad()
            optimiser_d_y.zero_grad()
            d_loss_real = ((d_x(x) - 1) **2).mean() + ((d_y(y) - 1)**2).mean()
            d_loss_fake = (d_y(fake_y.detach())**2).mean() + (d_x(fake_x.detach())**2).mean()
            total_d = (d_loss_real + d_loss_fake) * 0.5
            total_d.backward()
            optimiser_d_x.step()
            optimiser_d_y.step()
            optimiser_g_x_y.zero_grad()
            optimiser_g_y_x.zero_grad()
            loss_g_y_x = ((d_x(fake_x) - 1)**2).mean() + cycle_loss(y, g_x_y(fake_x))
            loss_g_x_y = ((d_y(fake_y) - 1)**2).mean() + cycle_loss(x, g_y_x(fake_y))
            loss_g_x_y.backward()
            loss_g_x_y.backward()
            optimiser_g_x_y.step()
            optimiser_g_y_x.step()
            total_d_loss += (total_d.item() * 2)
            total_g_x_y_loss += loss_g_x_y.item()
            total_g_y_x_loss += loss.g_y_x.item()
        avg_d_loss = total_d_loss / total_data
        avg_g_x_y_loss = total_g_x_y_loss / total_data
        avg_g_y_x_loss = total_g_y_x_loss / total_data
        if args.use_wandb:
            matrix_of_img = fake_y.detach()[:10,...]
            name = "examples " + "epoch " + str(epoch + 1)
            wandb.log({name : [wandb.Image(matrix_of_img[i]) for i in range(10)]})
            del matrix_of_img
            wandb.log({"Avg Discriminator loss": avg_d_loss, 'epoch': epoch + 1})
            wandb.log({"Avg Cartoon to Human loss": avg_g_x_y_loss, 'epoch': epoch + 1})
            wandb.log({"Avg Human to Cartoon loss": avg_g_y_x_loss, 'epoch': epoch + 1})
        print("Avg Discriminator Loss: {}".format(avg_d_loss))
        print("Avg Cartoon to Human Loss: {}".format(avg_g_x_y_loss))
        print("Avg Human to Cartoon Loss: {}".format(avg_g_y_x_loss))
    if args.use_wandb:
        torch.save(d_x.state_dict(), "d_x.pt")
        torch.save(d_y.state_dict(), "d_y.pt")
        torch.save(g_x_y.state_dict(), "g_x_y.pt")
        torch.save(g_y_x.state_dict(), "g_y_x.pt")
        wandb.save("d_x.pt")
        wandb.save("d_y.pt")
        wandb.save("g_x_y.pt")
        wandb.save("g_y_x.pt")

