import torch
import numpy as np
from torchvision import transforms
from torchmetrics.image import FID, SSIM
from torchvision import models
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

class ImageQualityAssessment:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.fid_metric = FID().to(self.device)
        self.ssim_metric = SSIM().to(self.device)
        self.inception_model = models.inception_v3(pretrained=True).to(self.device).eval()

    def calculate_fid(self, real_images, generated_images):
        """
        Calculates FID between real and generated images.
        """
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)

        # Resize images to 299x299 for Inception model input
        real_images_resized = self._resize_images(real_images)
        generated_images_resized = self._resize_images(generated_images)

        # Compute FID
        fid_score = self.fid_metric(real_images_resized, generated_images_resized)
        return fid_score.item()

    def calculate_ssim(self, real_images, generated_images):
        """
        Calculates SSIM between real and generated images.
        """
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)

        # Compute SSIM
        ssim_score = self.ssim_metric(real_images, generated_images)
        return ssim_score.item()

    def _resize_images(self, images):
        """
        Resize images to 299x299 for Inception model input.
        """
        resize_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        resized_images = torch.stack([resize_transform(image) for image in images])
        return resized_images

    def evaluate(self, real_images, generated_images):
        """
        Perform both FID and SSIM evaluations.
        """
        fid_score = self.calculate_fid(real_images, generated_images)
        ssim_score = self.calculate_ssim(real_images, generated_images)

        return {'FID': fid_score, 'SSIM': ssim_score}


# Example usage:
if __name__ == '__main__':
    # Set paths to your real and generated image directories
    real_image_dir = 'path/to/real_images'
    generated_image_dir = 'path/to/generated_images'

    # Load real and generated images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Assuming images are stored in folders (real and generated)
    real_dataset = ImageFolder(root=real_image_dir, transform=transform)
    generated_dataset = ImageFolder(root=generated_image_dir, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=False)

    # Initialize image quality assessment object
    evaluator = ImageQualityAssessment()

    # Loop through the datasets and compute FID and SSIM
    fid_total = 0
    ssim_total = 0
    count = 0

    for (real_batch, _), (generated_batch, _) in zip(real_loader, generated_loader):
        fid_score = evaluator.calculate_fid(real_batch, generated_batch)
        ssim_score = evaluator.calculate_ssim(real_batch, generated_batch)

        fid_total += fid_score
        ssim_total += ssim_score
        count += 1

    # Average scores
    avg_fid = fid_total / count
    avg_ssim = ssim_total / count

    print(f'Average FID: {avg_fid}')
    print(f'Average SSIM: {avg_ssim}')
