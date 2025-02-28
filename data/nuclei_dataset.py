import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from .utils.preprocessing import ImagePreprocessing  # Import the ImagePreprocessing class

class NucleiDataset(Dataset):
    def __init__(self, image_dir, metadata_csv, transform=None, augment=False):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            metadata_csv (str): Path to the CSV file containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool): Whether to apply augmentation on the images during training.
        """
        self.image_dir = image_dir
        self.metadata_df = pd.read_csv(metadata_csv)  # Load metadata from CSV
        self.transform = transform
        self.augment = augment

        # Instantiate the ImagePreprocessing class
        self.image_processor = ImagePreprocessing(image_size=(64, 64))

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

        # Apply augmentation only during training
        if self.augment:
            image = self.image_processor.random_augmentation(image)  # Apply random augmentation

        # Apply basic preprocessing (resize, normalize)
        image = self.image_processor.preprocess(image)

        return image, metadata


# Example usage:
if __name__ == '__main__':
    image_dir = 'path/to/your/images'
    metadata_csv = 'path/to/your/metadata.csv'

    # Define transformations (preprocessing only)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create the dataset with augmentation enabled (for training)
    train_dataset = NucleiDataset(image_dir=image_dir, metadata_csv=metadata_csv, transform=transform, augment=True)

    # Create DataLoader for batching
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Iterate through the dataloader
    for images, metadata in train_loader:
        print(images.shape)  # Batch of images (32, 3, 64, 64)
        print(metadata.shape)  # Corresponding metadata (32, number of metadata columns)
