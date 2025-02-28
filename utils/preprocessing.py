import random
import torch
from torchvision import transforms
from PIL import Image


class ImagePreprocessing:
    def __init__(self, image_size=(64, 64), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Define basic image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        # Define image augmentations
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def preprocess(self, image):
        """
        Apply basic preprocessing transformations (resize, to tensor, normalize).
        """
        return self.transform(image)

    def augment(self, image):
        """
        Apply random augmentations to the image (flip, rotate, color jitter).
        """
        return self.augmentations(image)

    def random_augmentation(self, image):
        """
        Apply random augmentation or no augmentation based on probability.
        """
        if random.random() > 0.5:  # 50% chance to apply augmentation
            return self.augment(image)
        return image


# Example usage:
if __name__ == '__main__':
    # Load an example image
    img_path = 'path/to/your/image.jpg'
    image = Image.open(img_path).convert('RGB')

    # Initialize the preprocessing class
    image_processor = ImagePreprocessing(image_size=(64, 64))

    # Preprocess the image (resize, normalize)
    preprocessed_image = image_processor.preprocess(image)
    print(f'Preprocessed Image Shape: {preprocessed_image.shape}')

    # Augment the image
    augmented_image = image_processor.random_augmentation(image)
    augmented_image.show()  # Display the augmented image
