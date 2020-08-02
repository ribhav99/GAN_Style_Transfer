import torch
import torch.nn as nn
import torch.optim as optim
from GeneratorAndDiscriminator import Generator, Discriminator
#from disAndGen3 import Generator, Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange
from datetime import datetime
from preprocess import make_human_text_file, make_cartoon_text_file
from train_test_split import split_human_data, split_cartoon_data


def train(args, device, wandb=None):

    # PREPROCESS DATA
    # make_human_text_file(args)
    # make_cartoon_text_file(args)
    # split_human_data(args)
    # split_cartoon_data(args)

    # LOAD DATA .......
    full_data = get_data_loader(args, train=True)

    d_x = Discriminator(args).to(device)  # x is cartoon
    d_y = Discriminator(args).to(device)  # y is human faces
    g_x_y = Generator(args).to(device)  # cartoon -> human faces
    g_y_x = Generator(args).to(device)  # human faces -> cartoon
    optimiser_d_x = optim.Adam(d_x.parameters(), lr=args.dis_learning_rate)
    optimiser_d_y = optim.Adam(d_y.parameters(), lr=args.dis_learning_rate)
    optimiser_g_x_y = optim.Adam(g_x_y.parameters(), lr=args.gen_learning_rate)
    optimiser_g_y_x = optim.Adam(g_y_x.parameters(), lr=args.gen_learning_rate)

    if args.load_models:
        print("Loading Models...")
        models = torch.load(args.model_path)
        d_x.load_state_dict(models['d_x'])
        d_y.load_state_dict(models['d_y'])
        g_x_y.load_state_dict(models['g_x_y'])
        g_y_x.load_state_dict(models['g_y_x'])
        optimiser_d_x.load_state_dict(models['optimiser_d_x'])
        optimiser_d_y.load_state_dict(models['optimiser_d_y'])
        optimiser_g_x_y.load_state_dict(models['optimiser_g_x_y'])
        optimiser_g_y_x.load_state_dict(models['optimiser_g_y_x'])
        print("Successfully Loaded Models...")
    if args.use_wandb:
        wandb.watch(d_x, log='all')
        wandb.watch(d_y, log='all')
        wandb.watch(g_x_y, log='all')
        wandb.watch(g_y_x, log='all')

    def cycle_loss(real, reconstructed):
        loss = (torch.abs(real - reconstructed)).mean()
        return args.lambda_cycle * loss

    d_x.train()
    d_y.train()
    g_x_y.train()
    g_y_x.train()

    if args.buffer_train:
        bufferX = torch.zeros(
            50, args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1]).to(device)
        bufferY = torch.zeros(
            50, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1]).to(device)
        samplesX = torch.zeros(
            args.batch_size, args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1]).to(device)
        samplesY = torch.zeros(
            args.batch_size, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1]).to(device)
        sampling = args.batch_size//2
    print("Start Training....")
    for epoch in trange(args.num_epochs):
        total_d_loss = 0.0
        total_g_x_y_loss = 0.0
        total_g_y_x_loss = 0.0
        total_data = 0

        if args.decay:
            dis_lr = args.dis_learning_rate - \
                ((args.dis_learning_rate / 100) * (epoch + 40))
            gen_lr = args.gen_learning_rate - \
                ((args.gen_learning_rate / 100) * (epoch + 40))

            for l in range(len(optimiser_d_x.param_groups)):
                optimiser_d_x.param_groups[l]['lr'] = dis_lr

            for l in range(len(optimiser_d_y.param_groups)):
                optimiser_d_y.param_groups[l]['lr'] = dis_lr

            for l in range(len(optimiser_g_x_y.param_groups)):
                optimiser_g_x_y.param_groups[l]['lr'] = gen_lr

            for l in range(len(optimiser_g_y_x.param_groups)):
                optimiser_g_y_x.param_groups[l]['lr'] = gen_lr

        for batch_num, data in enumerate(full_data):

            if args.buffer_train and batch_num == args.num_train_samples // args.batch_size:
                break  # last batch might not be full and therefore indexing errors

            y, x = data[0].to(device), data[1].to(
                device)  # x is cartoon, y is human
            total_data += x.shape[0]

            optimiser_g_x_y.zero_grad()
            optimiser_g_y_x.zero_grad()
            optimiser_d_x.zero_grad()
            optimiser_d_y.zero_grad()

            fake_x = g_y_x(y)
            fake_y = g_x_y(x)

            if args.buffer_train:
                if batch_num == 0 and epoch == 0:
                    bufferX = fake_x[:50].clone()
                    bufferY = fake_y[:50].clone()

                perm = torch.randperm(bufferX.size(0))
                idx = perm[:sampling]
                samplesX[:sampling] = bufferX[idx]
                bufferX[idx] = fake_x[sampling:]
                samplesX[sampling:] = fake_x[:sampling]

                samplesY[:sampling] = bufferY[idx]
                bufferY[idx] = fake_y[sampling:]
                samplesY[sampling:] = fake_y[:sampling]

                d_x_loss = ((d_x(x) - 1) ** 2).mean() + \
                    (d_x(samplesX.detach())**2).mean()
                d_y_loss = ((d_y(y) - 1) ** 2).mean() + \
                    (d_y(samplesY.detach())**2).mean()

            else:
                d_x_loss = ((d_x(x) - 1) ** 2).mean() + \
                    (d_x(fake_x.detach())**2).mean()
                d_y_loss = ((d_y(y) - 1) ** 2).mean() + \
                    (d_y(fake_y.detach())**2).mean()

                d_x_loss /= 2
                d_y_loss /= 2

            d_x_loss.backward()
            d_y_loss.backward()

            optimiser_d_x.step()
            optimiser_d_y.step()

            loss_g_y_x = ((d_x(fake_x) - 1)**2).mean() + \
                cycle_loss(y, g_x_y(fake_x))
            loss_g_x_y = ((d_y(fake_y) - 1)**2).mean() + \
                cycle_loss(x, g_y_x(fake_y))

            loss_g_y_x.backward()
            loss_g_x_y.backward()
            optimiser_g_x_y.step()
            optimiser_g_y_x.step()
            total_d_loss += d_x_loss.item() + d_y_loss.item()
            total_g_x_y_loss += loss_g_x_y.item()
            total_g_y_x_loss += loss_g_y_x.item()
        avg_d_loss = total_d_loss / total_data
        avg_g_x_y_loss = total_g_x_y_loss / total_data
        avg_g_y_x_loss = total_g_y_x_loss / total_data
        if args.use_wandb:
            if (epoch + 1) % args.image_save_f == 0:
                matrix_of_img = fake_y.detach()[:10, ...]
                name = "examples " + "epoch " + str(epoch + 1)
                wandb.log({name: [wandb.Image(matrix_of_img[i])
                                  for i in range(10)]})
                del matrix_of_img
            wandb.log({"Avg Discriminator loss": avg_d_loss, 'epoch': epoch + 1})
            wandb.log(
                {"Avg Cartoon to Human loss": avg_g_x_y_loss, 'epoch': epoch + 1})
            wandb.log(
                {"Avg Human to Cartoon loss": avg_g_y_x_loss, 'epoch': epoch + 1})
        print("Avg Discriminator Loss: {}".format(avg_d_loss))
        print("Avg Cartoon to Human Loss: {}".format(avg_g_x_y_loss))
        print("Avg Human to Cartoon Loss: {}".format(avg_g_y_x_loss))
    if args.use_wandb:
        time = datetime.now()
        torch.save({"d_x": d_x.state_dict(), "d_y": d_y.state_dict(), "g_x_y": g_x_y.state_dict(), "g_y_x": g_y_x.state_dict(), "optimiser_d_x": optimiser_d_x.state_dict(), "optimiser_d_y": optimiser_d_y.state_dict(), "optimiser_g_x_y": optimiser_g_x_y.state_dict(), "optimiser_g_y_x": optimiser_g_y_x.state_dict()},
                   'model{}.pt'.format(time))
        wandb.save('model{}.pt'.format(time))

    else:
        time = datetime.now()
        torch.save({"d_x": d_x.state_dict(), "d_y": d_y.state_dict(), "g_x_y": g_x_y.state_dict(), "g_y_x": g_y_x.state_dict(), "optimiser_d_x": optimiser_d_x.state_dict(), "optimiser_d_y": optimiser_d_y.state_dict(), "optimiser_g_x_y": optimiser_g_x_y.state_dict(), "optimiser_g_y_x": optimiser_g_y_x.state_dict()},
                   'args.save_path/' + 'model{}.pt'.format(time))
