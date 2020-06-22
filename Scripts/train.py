import torch
import torch.nn as nn
import torch.optim as optim
from GeneratorAndDiscriminator import Generator, Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange
from datetime import datetime
from preprocess import make_human_text_file, make_cartoon_text_file
from train_test_split import split_human_data, split_cartoon_data


def train(args, wandb=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preproess data
    # make_human_text_file(args)
    # make_cartoon_text_file(args)
    # split_human_data(args)
    # split_cartoon_data(args)

    # LOAD DATA .......
    full_data = get_data_loader(args, train=True)

    loss_function = nn.MSELoss()
    discriminator = Discriminator(args).to(device)
    generator = Generator(args).to(device)

    optimiser_d = optim.Adam(discriminator.parameters(),
                             lr=args.dis_learning_rate)
    optimiser_g = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)

    if args.use_wandb:
        wandb.watch(discriminator, log='all')
        wandb.watch(generator, log='all')

    discriminator.train()
    generator.train()

    print("Start Training....")
    for epoch in trange(args.num_epochs):
        loss_gen = 0.0
        loss_discrim = 0.0
        for batch_num, data in enumerate(full_data):
            human_faces, cartoon_faces = data
            batch_size = human_faces.shape[0]
            human_faces = human_faces.to(device)
            cartoon_faces = cartoon_faces.to(device)

            # Discriminator: max log(D(x)) + log(1 - D(G(z)))
            fake = generator(cartoon_faces).view(
                -1, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])

            if epoch % args.discrim_train_f == 0:
                optimiser_d.zero_grad()
                labels = torch.ones(batch_size, device=device)
                output_d = discriminator(
                    human_faces).view(-1, batch_size)  # here
                loss_d_real = loss_function(output_d, labels)

                labels = torch.zeros(batch_size, device=device)

                output_d = discriminator(
                    fake.detach()).view(-1, batch_size)  # here
                loss_d_fake = loss_function(output_d, labels)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimiser_d.step()
                loss_discrim += loss_d.item()

            # Generator: max log(D(G(z)))
            optimiser_g.zero_grad()
            labels = torch.ones(1, batch_size, device=device)
            output = discriminator(fake).view(-1, batch_size)  # here
            loss_g = loss_function(output, labels)
            loss_g.backward()
            optimiser_g.step()
            loss_gen += loss_g.item()

        if (epoch + 1) % args.image_save_f == 0:
            matrix_of_img = fake.detach()[:10, ...]
            if args.use_wandb:
                name = "examples " + "epoch " + str(epoch + 1)
                wandb.log({name: [wandb.Image(matrix_of_img[i])
                                  for i in range(10)]})
            else:
                # insert your plotting code here
                pass
            del matrix_of_img
        total = batch_num + 1
        avg_loss_gen = loss_gen / total
        avg_loss_discrim = loss_discrim / total
        if args.use_wandb:
            wandb.log(
                {"discriminator loss": avg_loss_discrim, 'epoch': epoch + 1})
            wandb.log({"generator loss": avg_loss_gen, 'epoch': epoch + 1})
        print("\nAverage discriminator loss for epoch {} is {}".format(
            epoch + 1, avg_loss_discrim))
        print("Average generator loss for epoch {} is {}".format(
            epoch + 1, avg_loss_gen))
        if not args.use_wandb:
            torch.save(discriminator.state_dict(), args.save_path +
                       "/{}disciminator-{}.pt".format(datetime.now(), epoch + 1))
            torch.save(generator.state_dict(), args.save_path +
                       "/{}generator-{}.pt".format(datetime.now(), epoch + 1))
    if args.use_wandb:
        torch.save(discriminator.state_dict(), "discriminator.pt")
        torch.save(generator.state_dict(), "generator.pt")
        wandb.save("discriminator.pt")
        wandb.save("generator.pt")
    else:
        torch.save(discriminator.state_dict(), args.save_path +
                   "/disciminator-{}-FINAL.pt".format(datetime.now()))
        torch.save(generator.state_dict(), args.save_path +
                   "/generator-{}-FINAL.pt".format(datetime.now()))


if __name__ == "__main__":
    ####
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
        'dis_learning_rate': 0.001,
        'gen_learning_rate': 0.002,
        'image_dimensions': (128, 128, 3),
        'cartoon_dimensions': (128, 128, 3),
        'batch_size': 64,
        'max_pool': (2, 2),
        'features_d': 16,
        'features_g': 16,
        'num_epochs': 30,
        'kernel_size': 3,
        'human_train_path': "/content/GAN_Style_Transfer/data/human_train.txt",
        'human_test_path': "/content/GAN_Style_Transfer/data/human_test.txt",
        'cartoon_train_path': "/content/GAN_Style_Transfer/data/cartoon_train.txt",
        'cartoon_test_path': "/content/GAN_Style_Transfer/data/cartoon_test.txt",
        'human_data_root_path': "/content/humanfaces128/",
        'cartoon_data_root_path': "/content/cartoonfaces/",
        'save_path': "/content/GAN_Style_Transfer/Models",
        'image_save_f': 1,  # i.e save an image every 1 epochs
        'discrim_train_f': 5,
        'use_wandb': True
    }
    args.update(args_dict)
    train(args)
