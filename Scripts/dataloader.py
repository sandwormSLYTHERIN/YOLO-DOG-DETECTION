# ----------------------------------------------------------
# Custom YOLO Dataloader with Class-Conditional Augmentation
# ----------------------------------------------------------

import numpy as np
import albumentations as A
from ultralytics.data.dataset import YOLODataset

class CustomYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define Albumentations augmentations for each class
        self.aug_majority = A.Compose([
            A.MotionBlur(p=0.05, blur_limit=(3, 5)),
            A.RandomBrightnessContrast(p=0.2),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),
        ])

        self.aug_minority = A.Compose([
            A.MotionBlur(p=0.2, blur_limit=(5, 9)),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussianNoise(p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.4),
        ])

    def load_image(self, i):
        # Use YOLO’s default image loading
        im, _ = super().load_image(i)

        # Identify class labels for the current image
        labels = self.labels[i][:, 0] if len(self.labels[i]) else []
        if len(labels) == 0:
            return im  # no augmentation if no labels

        # If any 'NonDog' class (assuming class 1 = NonDog)
        if 1 in labels:
            aug = self.aug_minority
        else:
            aug = self.aug_majority

        # Albumentations requires uint8
        im = np.ascontiguousarray(im)
        im_aug = aug(image=im)["image"]
        return im_aug
