import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class NucleiDataset(Dataset):
    def __init__(self, image_dir, metadata_csv, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            metadata_csv (str): Path to the CSV file containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.metadata_df = pd.read_csv(metadata_csv)  # Assuming the metadata is in a CSV file
        self.transform = transform

        # Assuming metadata CSV has columns like 'image_name' and 'metadata_1', 'metadata_2', etc.
        self.image_names = self.metadata_df['image_name'].values
        self.metadata_columns = [col for col in self.metadata_df.columns if col != 'image_name']

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # Get the image name and the metadata
        img_name = self.image_names[idx]
        metadata = self.metadata_df.iloc[idx, 1:].values  # All columns except 'image_name'
        metadata = torch.tensor(metadata, dtype=torch.float32)  # Convert metadata to tensor

        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, metadata


# Example usage:
if __name__ == '__main__':
    # Directory where images are stored
    image_dir = 'path/to/your/images'

    # Path to metadata CSV (which includes image names and corresponding metadata)
    metadata_csv = 'path/to/your/metadata.csv'

    # Define any image transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create the dataset
    dataset = NucleiDataset(image_dir=image_dir, metadata_csv=metadata_csv, transform=transform)

    # Create DataLoader for batching
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the dataloader
    for images, metadata in dataloader:
        print(images.shape)  # Batch of images (32, 3, 64, 64)
        print(metadata.shape)  # Corresponding metadata (32, number of metadata columns)
