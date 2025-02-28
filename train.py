import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from models.discriminator import Discriminator
from models.generator import Generator
from data import NucleiDataset  # Modify according to your dataset location

# Hyperparameters
batch_size = 32
image_shape = (3, 64, 64)  # Image shape (channels, height, width)
metadata_dim = 10  # Metadata size
latent_dim = 100  # Latent vector size
lr = 0.0002  # Learning rate
epochs = 100  # Number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data (replace with your actual data loading)
dataset = NucleiDataset()  # Define your dataset loading
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
discriminator = Discriminator(image_shape, metadata_dim).to(device)
generator = Generator(metadata_dim, latent_dim, image_shape).to(device)

# Optimizers
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for i, (real_images, real_metadata) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_metadata = real_metadata.to(device)

        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        discriminator.zero_grad()

        # Real images
        real_output = discriminator(real_images, real_metadata)
        d_loss_real = adversarial_loss(real_output, real_labels)

        # Fake images from Generator
        z = torch.randn(batch_size, latent_dim).to(device)  # Random noise
        fake_images = generator(z, real_metadata)  # Generator output
        fake_output = discriminator(fake_images.detach(), real_metadata)
        d_loss_fake = adversarial_loss(fake_output, fake_labels)

        # Total Discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        generator.zero_grad()

        # Fake images (for Generator's backprop)
        fake_output = discriminator(fake_images, real_metadata)
        g_loss = adversarial_loss(fake_output, real_labels)  # Want generator to fool the discriminator
        g_loss.backward()
        optimizer_g.step()

        # Print progress
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Optionally save generated images or models after every epoch
    # torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
    # torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

