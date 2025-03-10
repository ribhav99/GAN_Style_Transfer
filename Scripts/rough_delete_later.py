import torch
import torch.nn as nn
import torch.optim as optim
from GeneratorAndDiscriminator import Generator, Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange
from datetime import datetime

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #LOAD DATA .......
    # full_data = get_data_loader(args, train = True)

    loss_function = nn.BCELoss()
    discriminator = Discriminator(args.image_dimensions, args.features_d, args.kernel_size).to(device)
    generator = Generator(args.cartoon_dimensions, args.image_dimensions, features_g=args.features_g, kernel_size=args.kernel_size, max_pool=args.max_pool).to(device)

    optimiser_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
    optimiser_g = optim.Adam(generator.parameters(), lr=args.learning_rate)

    discriminator.train()
    generator.train()

    print("Start Training....")
    for epoch in trange(args.num_epochs):
        # for batch_num, data in enumerate(full_data):
        # human_faces, cartoon_faces = data
        human_faces = torch.rand(args.image_dimensions).view(-1, 3, args.image_dimensions[0], args.image_dimensions[1])
        cartoon_faces = torch.rand(args.cartoon_dimensions).view(-1, args.cartoon_dimensions[2], args.cartoon_dimensions[0], args.cartoon_dimensions[1])

        #Discriminator: max log(D(x)) + log(1 - D(G(z)))
        optimiser_d.zero_grad()
        labels = torch.ones(args.batch_size, device = device)
        output_d = discriminator(human_faces).reshape(-1)
        loss_d_real = loss_function(output_d, labels)
        
        fake = generator(cartoon_faces).view(-1, args.image_dimensions[2], args.image_dimensions[0], args.image_dimensions[1])
        labels = torch.zeros(args.batch_size, device = device)

        output_d = discriminator(fake.detach()).reshape(-1)
        loss_d_fake = loss_function(output_d, labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimiser_d.step()

        #Generator: max log(D(G(z)))
        optimiser_g.zero_grad()
        labels = torch.ones(args.batch_size, device = device)
        output = discriminator(fake).reshape(-1)
        loss_g = loss_function(output, labels)
        loss_g.backward()
        optimiser_g.step()

            #check training progress
            #can do it per batch size or per epoch
            #if per epoch then it comes after this for loop
            # if batch_num % 300 == 0:
            #     print("accuracy")
    torch.save(discriminator, "disciminator-{}.pt".format(datetime.now()))
    torch.save(generator, "generator-{}.pt".format(datetime.now()))

if __name__ == "__main__":
    #these are the ones we have to change most likely
    root_human_data = "/Users/gerald/Desktop/GAN datasets/humanfaces"
    root_cartoon_data = "/Users/gerald/Desktop/GAN datasets/cartoonfaces"
    ####
    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
        'learning_rate': 0.001,
        'batch_size' : 1,
        'image_dimensions' : (218, 178, 3), 
        'cartoon_dimensions' : (128, 128, 3),
        'max_pool' : (2, 2),
        'features_d' : 64,
        'features_g' : 2,
        'num_epochs' : 1,
        'kernel_size' : 3,
        'human_train_path' : "../data/human_train.txt",
        'human_test_path' : "../data/human_test.txt",
        'cartoon_train_path' : "../data/cartoon_train.txt",
        'cartoon_test_path' : "../data/cartoon_test.txt",
        'human_data_root_path' : root_human_data,
        'cartoon_data_root_path' : root_cartoon_data
    }
    args.update(args_dict)
    train(args)
