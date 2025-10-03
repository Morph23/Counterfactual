"""
Data loading utilities for image counterfac        print(f"Loading {len(cat_files)} cat images...")ual explanations.
"""

import numpy as np
import os
from pathlib import Path
import cv2
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Data loader for image counterfactual experiments."""
    
    def load_kaggle_cats_dogs(self, data_path="data/kaggle_cats_dogs", 
                             max_samples_per_class=500, target_size=(64, 64)):
        """Load Kaggle cats vs dogs dataset with real images.
        
        Dataset: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
        
        Expected structure:
        data_path/
        ├── Cat/
        │   ├── 1.jpg, 2.jpg, ...
        └── Dog/
            ├── 1.jpg, 2.jpg, ...
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {data_path}. "
                "Download the Kaggle Cats vs Dogs dataset and place it under data/kaggle_cats_dogs/."
            )
        
        images = []
        labels = []
        
        # Load Cat images (label = 0)
        cat_path = data_path / 'Cat'
        if not cat_path.exists():
            raise FileNotFoundError(f"Cat folder not found at {cat_path}")

        cat_files = list(cat_path.glob('*.jpg'))[:max_samples_per_class]
        if len(cat_files) == 0:
            raise ValueError(f"No cat images found in {cat_path}")

        print(f"� Loading {len(cat_files)} cat images...")

        for img_file in cat_files:
            img = cv2.imread(str(img_file))
            if img is None:
                warnings.warn(f"Unable to read image: {img_file.name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(0)  # Cat = 0
        
        # Load Dog images (label = 1)
        dog_path = data_path / 'Dog'
        if not dog_path.exists():
            raise FileNotFoundError(f"Dog folder not found at {dog_path}")

        dog_files = list(dog_path.glob('*.jpg'))[:max_samples_per_class]
        if len(dog_files) == 0:
            raise ValueError(f"No dog images found in {dog_path}")

        print(f"Loading {len(dog_files)} dog images...")

        for img_file in dog_files:
            img = cv2.imread(str(img_file))
            if img is None:
                warnings.warn(f"Unable to read image: {img_file.name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(1)  # Dog = 1
        
        if len(images) == 0:
            raise RuntimeError("No images were loaded; ensure the dataset contains valid JPG files.")
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=int)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        print(f"Real dataset loaded: {len(images)} images")
        print(f"Distribution - Cats: {np.sum(labels == 0)}, Dogs: {np.sum(labels == 1)}")
        
        return images, labels
    
    def preprocess_images(self, images, target_size=None, normalize=True):
        """Preprocess images for model input."""
        processed = images.copy()
        
        if target_size is not None:
            resized = []
            for img in processed:
                resized_img = cv2.resize(img, target_size)
                resized.append(resized_img)
            processed = np.array(resized)
        
        if normalize and processed.max() > 1.0:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def save_dataset(self, images, labels, name="dataset"):
        """Save dataset to disk."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        np.save(data_dir / f"{name}_images.npy", images)
        np.save(data_dir / f"{name}_labels.npy", labels)
        
        print(f"Dataset saved: {data_dir}/{name}_*.npy")
        return str(data_dir / name)
    
    def load_dataset(self, name="dataset"):
        """Load dataset from disk."""
        data_dir = Path("data")
        
        images_path = data_dir / f"{name}_images.npy"
        labels_path = data_dir / f"{name}_labels.npy"
        
        if not (images_path.exists() and labels_path.exists()):
            raise FileNotFoundError(f"Dataset files not found: {images_path}, {labels_path}")
        
        images = np.load(images_path)
        labels = np.load(labels_path)
        
        print(f"Dataset loaded: {images.shape}, {labels.shape}")
        return images, labels
