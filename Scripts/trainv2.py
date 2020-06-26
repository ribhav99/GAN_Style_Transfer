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

    discriminator = Discriminator(args).to(device)
    generator = Generator(args).to(device)
    optimiser_d = optim.Adam(discriminator.parameters(), lr=args.dis_learning_rate)
    optimiser_g = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)
    dis_schedule = optim.lr_scheduler.CosineAnnealingLR(optimiser_d, args.num_epochs)
    gen_schedule = optim.lr_scheduler.CosineAnnealingLR(optimiser_g, args.num_epochs)
    if args.use_wandb:
        wandb.watch(discriminator, log='all')
        wandb.watch(generator, log = 'all')

    discriminator.train()
    generator.train()

    print("Start Training....")
    for epoch in trange(args.num_epochs):
        loss_gen = 0.0
        loss_discrim = 0.0
        total_discrim_trained = 0
        total_gen_trained = 0
        for batch_num, data in enumerate(full_data):
            human_faces, cartoon_faces = data
            batch_size = human_faces.shape[0]
            fake_images = generator(cartoon_faces)
            to_train_dis = (batch_num % 3 == 0 or batch_num % 3 == 1) #train discrim twice before gen trains
            if to_train_dis:
                optimiser_d.zero_grad()
                real_pred = discriminator(human_faces)
                fake_pred = discriminator(fake_images.detach())
                loss_d = ((real_pred - 1)**2).mean() + (fake_pred**2).mean()
                loss_d.backward()
                optimiser_d.step()
                loss_discrim += loss_d.item()
                total_discrim_trained += 1
            else:
                optimiser_g.zero_grad()
                fake_pred_for_generator = discriminator(fake_images)
                loss_g = ((fake_pred_for_generator - 1)**2).mean() 
                loss_g.backward()
                optimiser_g.step()
                loss_gen += loss_g.item()
                total_gen_trained += 1
        dis_schedule.step()
        gen_schedule.step()
        if (epoch + 1) % args.image_save_f == 0: 
            matrix_of_img = fake_images.detach()[:10,...]
            if args.use_wandb:
                name = "examples " + "epoch " + str(epoch + 1)
                wandb.log({name : [wandb.Image(matrix_of_img[i]) for i in range(10)]})
            del matrix_of_img
        if total_discrim_trained != 0:
            avg_loss_discrim = loss_discrim / total_discrim_trained
            wandb.log({"discriminator loss": avg_loss_discrim, 'epoch': epoch + 1})
            print("\nAverage discriminator loss for epoch {} is {}".format(epoch + 1, avg_loss_discrim))
        if total_gen_trained != 0:
            avg_loss_gen = loss_gen / total_gen_trained
            wandb.log({"generator loss": avg_loss_gen, 'epoch': epoch + 1})
            print("Average generator loss for epoch {} is {}".format(epoch + 1, avg_loss_gen)) 
    if args.use_wandb:
        torch.save(discriminator.state_dict(), "discriminator.pt")
        torch.save(generator.state_dict(), "generator.pt")
        wandb.save("discriminator.pt")
        wandb.save("generator.pt")

if __name__ == "__main__":
    #these are the ones we have to change most likely
    root_human_data = "/Users/gerald/Desktop/GAN datasets/humanfaces"
    root_cartoon_data = "/Users/gerald/Desktop/GAN datasets/cartoonfaces"
    ####
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
    'dis_learning_rate': 0.001,
    'gen_learning_rate': 0.001,
    'batch_size' : 64,
    'num_epochs' : 30,
    'human_train_path' : "/content/GAN_Style_Transfer/data/human_train.txt",
    'human_test_path' : "/content/GAN_Style_Transfer/data/human_test.txt",
    'cartoon_train_path' : "/content/GAN_Style_Transfer/data/cartoon_train.txt",
    'cartoon_test_path' : "/content/GAN_Style_Transfer/data/cartoon_test.txt",
    'human_data_root_path' : "/content/humanfaces/",
    'cartoon_data_root_path' : "/content/cartoonfaces/",
    'image_save_f' : 1, #i.e save an image every 10 epochs
    'act_fn' : 'relu',
    'pool_type' : 'max',
    'use_wandb' : False
    }
    args.update(args_dict)
    train(args)
