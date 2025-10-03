"""
Quick test script to verify counterfactual generation works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import DataLoader
from src.models.cnn_classifier import CNNClassifier
from src.counterfactuals.gradient_based import GradientBasedCounterfactuals
import numpy as np

print("="*60)
print("COUNTERFACTUAL GENERATION TEST")
print("="*60)

# Load dataset
print("\n[1/3] Loading dataset...")
loader = DataLoader()
images, labels = loader.load_kaggle_cats_dogs(
    data_path='data/kaggle_cats_dogs',
    max_samples_per_class=100,  # Smaller for quick test
    target_size=(64, 64)
)
print(f"Loaded {len(images)} images")

# Train a simple model
print("\n[2/3] Training model (quick, 5 epochs)...")
classifier = CNNClassifier(input_shape=(64, 64, 3))
model = classifier.build_model()
classifier.train(images, labels, epochs=5, verbose=0)
print("Model trained")

# Test counterfactual on a correctly classified image
print("\n[3/3] Testing counterfactual generation...")

# Find an image that's correctly classified
test_results = []
for i in range(min(20, len(images))):
    pred_class = classifier.predict_class(np.expand_dims(images[i], 0))[0]
    true_label = labels[i]
    
    if pred_class == true_label:
        print(f"\n[FOUND] Correctly classified image at index {i}")
        print(f"   True label: {true_label} ({'Cat' if true_label == 0 else 'Dog'})")
        print(f"   Model prediction: {pred_class} ({'Cat' if pred_class == 0 else 'Dog'})")
        
        # Generate counterfactual
        cf_generator = GradientBasedCounterfactuals(model)
        target = 1 - pred_class
        
        print(f"   Generating counterfactual to flip to class {target} ({'Cat' if target == 0 else 'Dog'})...")
        
        result = cf_generator.generate_counterfactual(
            images[i],
            target_class=target,
            max_iterations=300,
            verbose=False
        )
        
        print(f"\n   Result:")
        print(f"      Original: class {result['original_class']} (conf: {result['original_confidence']:.3f})")
        print(f"      Final: class {result['final_class']} (conf: {result['final_confidence']:.3f})")
        print(f"      Success: {'YES' if result['success'] else 'NO'}")
        print(f"      Proximity: {result['proximity']:.4f}")
        print(f"      Iterations: {result['iterations']}")
        
        test_results.append(result)
        
        if len(test_results) >= 3:
            break

print(f"\n{'='*60}")
print(f"SUMMARY: {sum(r['success'] for r in test_results)}/{len(test_results)} counterfactuals succeeded")
print(f"{'='*60}")
