import torch
from torchvision.utils import save_image
from models.generator import Generator
from utils.preprocessing import ImagePreprocessing
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = 'generated_images'  # Directory to save generated images
os.makedirs(output_dir, exist_ok=True)  # Make output directory if it doesn't exist

# Load the pre-trained generator model
generator = Generator().to(device)
generator.load_state_dict(torch.load('path/to/your/trained_generator.pth'))  # Path to trained model

# Optionally, you can load metadata if you want to condition the generator (e.g., clinical features)
metadata_csv = 'path/to/your/metadata.csv'
metadata_df = pd.read_csv(metadata_csv)

# Helper function to generate random noise
def generate_random_noise(batch_size, z_dim=100):
    return torch.randn(batch_size, z_dim).to(device)

# Helper function to generate images
def generate_images(generator, batch_size, z_dim=100, metadata=None):
    noise = generate_random_noise(batch_size, z_dim)
    if metadata is not None:
        # Optionally condition the generator with metadata (e.g., clinical features)
        metadata = metadata.to(device)
        generated_images = generator(noise, metadata)
    else:
        # If no metadata, generate images without conditioning
        generated_images = generator(noise)
    
    return generated_images

# Generate a batch of images
batch_size = 8
z_dim = 100  # Latent space dimension (adjust as needed)

# Optionally, use metadata from the dataset if conditioning is enabled
# Here, we assume the metadata is the same size as the batch size for simplicity
metadata_batch = torch.tensor(metadata_df.iloc[:batch_size, 1:].values, dtype=torch.float32)

# Generate images (with or without metadata conditioning)
generated_images = generate_images(generator, batch_size, z_dim, metadata=metadata_batch)

# Save the generated images
for i, img in enumerate(generated_images):
    img_path = os.path.join(output_dir, f"generated_img_{i+1}.png")
    save_image(img, img_path, normalize=True)  # Normalize to [0, 1] and save image

print(f"Generated images saved to {output_dir}")
