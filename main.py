"""
Main entry point for the Counterfactual Explanations project.
Demonstrates the complete workflow with real Kaggle images.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import DataLoader
from src.models.cnn_classifier import CNNClassifier
from src.counterfactuals.gradient_based import GradientBasedCounterfactuals
import numpy as np


def main():
    """Main execution function."""
    
    print("=" * 60)
    print("IMAGE COUNTERFACTUAL EXPLANATIONS")
    print("Cats vs Dogs Classification with Real Images")
    print("=" * 60)
    
    # 1. Load Real Dataset
    print("\n[1/4] Loading Kaggle Cats vs Dogs Dataset...")
    loader = DataLoader()
    try:
        images, labels = loader.load_kaggle_cats_dogs(
            data_path='data/kaggle_cats_dogs',
            max_samples_per_class=500,
            target_size=(64, 64)
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset")
        print("\nAnd place it in: data/kaggle_cats_dogs/")
        return
    
    print(f"Loaded {len(images)} images")
    print(f"   - Cats: {np.sum(labels == 0)}")
    print(f"   - Dogs: {np.sum(labels == 1)}")
    
    # 2. Train CNN Classifier
    print("\n[2/4] Training CNN Classifier...")
    classifier = CNNClassifier(input_shape=(64, 64, 3))
    model = classifier.build_model()
    
    # Train with a subset for faster demo
    train_size = min(len(images), 1000)
    history = classifier.train(
        images[:train_size], 
        labels[:train_size],
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # 3. Generate Counterfactuals
    print("\n[3/4] Generating Counterfactuals...")
    cf_generator = GradientBasedCounterfactuals(model)
    
    # Select a test image (first cat image)
    test_idx = np.where(labels == 0)[0][0]
    test_image = images[test_idx]
    test_label = labels[test_idx]
    
    # Get model's prediction on the original image
    pred = classifier.predict(np.expand_dims(test_image, 0))[0]
    original_pred_class = classifier.predict_class(np.expand_dims(test_image, 0))[0]
    
    print(f"\nTest Image Info:")
    print(f"   True Label: {'Cat' if test_label == 0 else 'Dog'} (class {test_label})")
    print(f"   Model Prediction: {'Cat' if original_pred_class == 0 else 'Dog'} (class {original_pred_class}, confidence: {pred if original_pred_class == 1 else 1-pred:.3f})")
    
    # Target is the opposite of what the model predicts
    target_class = 1 - original_pred_class
    print(f"   Target for Counterfactual: {'Cat' if target_class == 0 else 'Dog'} (class {target_class})")
    
    # Generate counterfactual
    result = cf_generator.generate_counterfactual(
        test_image,
        target_class=target_class,
        max_iterations=500,
        verbose=True
    )
    
    # 4. Evaluate Results
    print("\n[4/4] Evaluation Results:")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Original Prediction: {'Cat' if result['original_class'] == 0 else 'Dog'} ({result['original_confidence']:.3f})")
    print(f"Counterfactual Prediction: {'Cat' if result['cf_class'] == 0 else 'Dog'} ({result['cf_confidence']:.3f})")
    print(f"Proximity (L2): {result['proximity']:.4f}")
    print(f"Sparsity (L1): {result['sparsity']:.4f}")
    print(f"Iterations: {result['iterations']}")
    print("=" * 60)
    
    # Batch evaluation
    print("\nGenerating batch evaluation...")
    test_images = images[100:105]  # 5 test images
    results = []
    
    for i, img in enumerate(test_images):
        pred_class = classifier.predict_class(np.expand_dims(img, 0))[0]
        target = 1 - pred_class  # Flip class
        
        result = cf_generator.generate_counterfactual(
            img,
            target_class=target,
            max_iterations=300,
            verbose=False
        )
        results.append(result)
    
    # Evaluate batch
    evaluation = cf_generator.evaluate_counterfactuals(results, verbose=True)
    
    print("\nCounterfactual generation complete!")
    print("\n[TIP] Use visualization tools in src/utils/visualization.py")
    print("   to create visual comparisons of counterfactuals.")


if __name__ == "__main__":
    main()
