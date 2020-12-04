import os
import sys
from itertools import count
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size = 512
image_dim = 64

''' Notes
See https://github.com/soumith/ganhacks for GAN tips
https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
'''

# Run name
run_name = 'ttur_bs512_lblsmooth_discNoise'

# Number of image channels
num_channels = 3

# Size of input noise
num_z = 100

# Number of generator features (scaled)
ngf = 64

# Number of discriminator features (scaled)
ndf = 64

# Number of training epochs
num_epochs = 1000

# Learning rate
lr_g = 0.0001
lr_d = 0.0004

# Beta hparam for Adam
beta1 = 0.5

# Dataset and Dataloader
dataroot = "./data"
num_workers = 4

# Progress Picture Frequency
progress_pic_freq = 100

# Discriminator Noise
noise_disc_std = 0.02

save_file = './iter22900.save'

# Initialize Conv and BatchNorm layers from uniform distribution
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_z, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # Create dataset and dataloader
    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((64, 64)),  # Calculated for 144p video to 64 conversion
                                                   transforms.CenterCrop(image_dim),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Setup GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Plot some images
    display_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('Sample Train Images')
    plt.imshow(np.transpose(torchvision.utils.make_grid(display_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()

    # Create Generator
    netG = Generator().to(device)

    # Create Discriminator
    netD = Discriminator().to(device)

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function
    loss_fn = nn.BCELoss()

    # Fixed noise vectors (To visualize training progress)
    fixed_noise = torch.randn(64, num_z, 1, 1, device=device)

    # Real and Fake labels
    real_label = 1  # Actual image in dataset
    fake_label = 0  # Image created by Generator
    label_smoothing_alpha = 0.1  # When training discriminator, the target for real images becomes (real_label - label_smoothing_alpha)

    # Optimizers
    optimD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    batch_iters = 0
    saved_epoch = 0
    # Load saved model
    if save_file is not None and save_file != "":
        save = torch.load(save_file)
        optimG.load_state_dict(save['optimG_params'])
        optimD.load_state_dict(save['optimD_params'])
        netG.load_state_dict(save['gen_params'])
        netD.load_state_dict(save['disc_params'])
        saved_epoch = save['epoch']
        batch_iters = save['batch_iters']
        print(f"Resuming training from epoch {saved_epoch} and batch_iters {batch_iters}")


    # Tensorboard for logging
    logger = SummaryWriter()

    # Log net architecture
    logger.add_text("net_architecture/Generator", str(netG))
    logger.add_text("net_architecture/Discriminator", str(netD))

    # Create model checkpoint directory
    for i in count():
        if not os.path.exists(f'./saves/{run_name}-{i}'):
            run_name = f'{run_name}-{i}'
            break
    Path(f'./saves/{run_name}').mkdir(parents=True, exist_ok=True)
    save_path = f'./saves/{run_name}'

    # ================= Training Loop ================= #
    print("Starting Training...")


    for epoch in range(saved_epoch, num_epochs):
        for batch_num, batch_data in enumerate(dataloader, 0):

            # ----- Update Discriminator ----- #
            netD.zero_grad()

            # Train all real batch
            real_data = batch_data[0].to(device)
            cur_batch_size = real_data.size(0)
            labels = torch.full((cur_batch_size,), (real_label - label_smoothing_alpha), dtype=torch.float, device=device)

            discriminator_noise = torch.randn_like(real_data, device=device) * noise_disc_std
            real_data = real_data + discriminator_noise

            output = netD(real_data).view(-1)  # Forward pass and unroll
            lossD_real = loss_fn(output, labels)  # Compute loss for real data
            lossD_real.backward()  # Accumulate gradients

            logger.add_scalar("pred_val/D_real_avg", output.mean().item(), batch_iters)  # Log avg prediction for real images

            # Train with all fake batch
            noise = torch.randn(cur_batch_size, num_z, 1, 1, device=device)  # Generate random inputs to pass to G
            fake_data = netG(noise)  # Generate fake images
            labels.fill_(fake_label)  # Create labels

            discriminator_noise = torch.randn_like(fake_data, device=device) * noise_disc_std
            fake_data = fake_data + discriminator_noise

            output = netD(fake_data.detach()).view(-1)  # Detach, predict, and unroll
            lossD_fake = loss_fn(output, labels) # Compute loss for fake images
            lossD_fake.backward()  # Accumulate gradients

            optimD.step()  # Update D network parameters

            logger.add_scalar("pred_val/D_fake_avg", output.mean().item(), batch_iters)  # Log avg prediction for fake images
            logger.add_scalar("Loss/D_real_loss", lossD_real.detach(), batch_iters)
            logger.add_scalar("Loss/D_fake_loss", lossD_fake.detach(), batch_iters)
            logger.add_scalar("Loss/D_total_loss", (lossD_real + lossD_fake).detach(), batch_iters)

            # ----- Update Generator ----- #
            netG.zero_grad()

            labels.fill_(real_label)  # Labels are opposite when training G since we want D to be wrong
            output = netD(fake_data).view(-1)  # Pass fake_data through D again since D was updated TODO is this necessary?
            lossG = loss_fn(output, labels)  # Compute loss for generator based on D output
            lossG.backward()  # Compute gradients
            optimG.step()  # Update generator

            logger.add_scalar("pred_val/D_postUpdate_fake_avg", output.mean().item(), batch_iters)
            logger.add_scalar("Loss/G_loss", lossG, batch_iters)

            # ----- Run Generator on fixed inputs ----- #
            # Run every 500th batch and on last batch of last epoch
            if (batch_iters % progress_pic_freq == 0) or ((epoch == num_epochs) and (batch_num == len(dataloader) - 1)):

                # Generate images
                with torch.no_grad():
                    images = netG(fixed_noise).detach().cpu()

                # Convert images to grid
                grid_img = torchvision.utils.make_grid(images, padding=2, normalize=True)

                # Save images
                torchvision.utils.save_image(grid_img, f'{save_path}/iter{batch_iters}.png')
                logger.add_image("fixed_inputs", grid_img, batch_iters)

                # Checkpoint model
                try:
                    torch.save({
                        'epoch': epoch,
                        'batch_iters': batch_iters,
                        'optimG_params': optimG.state_dict(),
                        'optimD_params': optimD.state_dict(),
                        'gen_params': netG.state_dict(),
                        'disc_params': netD.state_dict(),
                        'seed': torch.seed(),
                    }, f'{save_path}/iter{batch_iters}.save')
                except:
                    print("Error checkpointing model")

                # Print progress msg
                print(f'Epoch: {epoch}/{num_epochs}\tbatch_prog: {batch_num}/{len(dataloader)}\titer: {batch_iters}')

            # Increment batch_iters
            batch_iters += 1







