import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=10, img_size=128):
        """
        Generator for Conditional GAN using transposed convolutions.
        :param latent_dim: Size of the noise vector (z)
        :param condition_dim: Size of the conditioning vector (c)
        :param img_size: Output image size (assumes square)
        """
        super(Generator, self).__init__()
        self.img_size = img_size
        input_dim = latent_dim + condition_dim  # Concatenate noise + condition

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU()
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.Tanh()  # Normalize output between -1 and 1
        )

    def forward(self, z, c):
        """
        Forward pass of the Generator.
        :param z: Random noise tensor (batch_size, latent_dim)
        :param c: Condition vector (batch_size, condition_dim)
        :return: Generated nuclei image (batch_size, 1, img_size, img_size)
        """
        x = torch.cat((z, c), dim=1)  # Concatenate noise and condition
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)  # Reshape to (batch, 512, 8, 8)
        img = self.conv_blocks(x)
        return img  # Shape: (batch, 1, 128, 128)

# Visualization function
def visualize_generated_images(generator, num_images=16, latent_dim=100, condition_dim=10, img_size=128):
    """
    Generates and visualizes synthetic nuclei images using the trained Generator.
    """
    generator.eval()  # Set to evaluation mode

    noise = torch.randn(num_images, latent_dim)  # Generate random noise
    condition = torch.randint(0, 2, (num_images, condition_dim)).float()  # Random binary conditioning labels

    with torch.no_grad():  # Disable gradient computation for inference
        fake_images = generator(noise, condition)

    fake_images = fake_images.cpu().numpy().squeeze()  # Convert to NumPy array
    fake_images = (fake_images + 1) / 2  # Rescale from [-1,1] to [0,1]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i], cmap="gray")
        ax.axis("off")

    plt.suptitle("Generated Nuclei Images")
    plt.show()

# Example Usage
if __name__ == "__main__":
    batch_size = 16
    latent_dim = 100
    condition_dim = 10
    img_size = 128

    # Initialize Generator
    generator = Generator(latent_dim, condition_dim, img_size)

    # Test forward pass
    noise = torch.randn(batch_size, latent_dim)  # Random noise vector
    condition = torch.randint(0, 2, (batch_size, condition_dim)).float()  # Random condition labels

    fake_images = generator(noise, condition)
    print("Generated Image Shape:", fake_images.shape)  # Expected: (16, 1, 128, 128)

    # Visualize generated images
    visualize_generated_images(generator)
