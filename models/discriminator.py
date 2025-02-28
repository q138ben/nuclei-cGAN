import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_shape, metadata_dim):
        super(Discriminator, self).__init__()
        
        # Image input processing layers (convolutional layers)
        self.conv1 = nn.Conv2d(image_shape[0], 64, kernel_size=3, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()

        # Metadata input processing layers (fully connected layers)
        self.dense_metadata = nn.Linear(metadata_dim, 256)
        self.leaky_relu_metadata = nn.LeakyReLU(0.2)
        self.batch_norm_metadata = nn.BatchNorm1d(256)

        # Fully connected layers after concatenation
        self.dense1 = nn.Linear(256 + 256, 512)
        self.leaky_relu4 = nn.LeakyReLU(0.2)
        self.batch_norm4 = nn.BatchNorm1d(512)

        # Output layer (sigmoid activation)
        self.output_layer = nn.Linear(512, 1)

    def forward(self, image_input, metadata_input):
        # Process image input
        x = self.conv1(image_input)
        x = self.leaky_relu1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.batch_norm3(x)

        x = self.flatten(x)

        # Process metadata input
        metadata = self.dense_metadata(metadata_input)
        metadata = self.leaky_relu_metadata(metadata)
        metadata = self.batch_norm_metadata(metadata)

        # Concatenate image and metadata features
        combined = torch.cat([x, metadata], dim=1)

        # Fully connected layers after concatenation
        x = self.dense1(combined)
        x = self.leaky_relu4(x)
        x = self.batch_norm4(x)

        # Output layer (sigmoid activation)
        output = torch.sigmoid(self.output_layer(x))

        return output


# Example usage
if __name__ == "__main__":
    image_shape = (3, 64, 64)  # Example image shape (channels, height, width)
    metadata_dim = 10  # Example metadata dimension

    # Instantiate the Discriminator
    discriminator = Discriminator(image_shape, metadata_dim)

    # Example inputs
    fake_images = torch.randn(1, *image_shape)  # Batch of fake images (1, 3, 64, 64)
    fake_metadata = torch.randn(1, metadata_dim)  # Example metadata (1, 10)

    # Forward pass
    output = discriminator(fake_images, fake_metadata)
    print("Discriminator output:", output)
