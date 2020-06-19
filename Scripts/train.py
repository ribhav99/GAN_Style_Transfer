import torch
import torch.nn as nn
import torch.optim as optim
from GeneratorAndDiscriminator import Generator, Discriminator
from tqdm import tqdm, trange

learning_rate = 0.001
batch_size = 32
image_dimensions = (218, 178, 3)
cartoon_dimensions = (128, 128, 3)
features_d = 64
features_g = 2
num_epochs = 30
kernel_size = 3
loss_function = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOAD DATA .......
human_faces_dataloader = ''
cartoon_faces_dataloader = ''

disciminator = Discriminator(features_d, kernel_size).to(device)
generator = Generator(features_g, kernel_size).to(device)

optimiser_d = optim.Adam(disciminator.parameters(), lr=learning_rate)
optimiser_g = optim.Adam(generator.parameters(), lr=learning_rate)

disciminator.train()
generator.train()

print("Start Training....")
for epoch in trange(num_epochs):
    for batch_num, data in enumerate(human_faces_dataloader):
        data = data.to(device)

        #Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disciminator.zero_grad()
        labels = torch.ones(batch_size).to_device
        output_d = disciminator(data).reshape(-1)
        loss_d_real = loss_function(output_d, labels)
        
        cartoons = cartoon_faces_dataloader[batch_num * batch_size: (batch_num * batch_size )+ batch_size]
        fake = generator(cartoons)
        labels = torch.zeros(batch_size).to(device)

        output_d = discriminator(fake.detach()).reshape(-1)
        loss_d_fake = loss_function(output_d, labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimiser_d.step()

        #Generator: max log(D(G(z)))
        generator.zero_grad()
        labels = torch.ones(batch_size).to(device)
        output = discriminator(fake).reshape(-1)
        loss_g = loss_function(output, labels)
        loss_g.backward()
        optimiser_g.step()

        #check training progress
        #can do it per batch size or per epoch
        #if per epoch then it comes after this for loop
        if batch_num % 300 == 0:
            print("accuracy")



