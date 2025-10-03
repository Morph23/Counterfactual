# Image Counterfactual Explanations for Deep Learning

An explainable AI project that generates counterfactual explanations for image classification models using **real Kaggle dataset images** (cats vs dogs).

## Project Overview

This project answers the question: **"What minimal pixel changes would flip an image prediction?"**

### Example:
**For a real cat image classified as "cat," show exactly which pixels need to change to make the model predict "dog"**

![Counterfactual Example](assets/counterfactual_example.png)

*Visual example showing: Original Image (Cat) → Counterfactual (Dog) → Perturbation Heatmap*

## Features

- **CNN Image Classifier**: Deep learning model for cats vs dogs binary classification
- **Real Dataset Only**: Uses Kaggle cats vs dogs dataset with real images (no synthetic fallbacks)
- **Gradient-based Counterfactuals**: Find minimal pixel perturbations that flip predictions
- **Comprehensive Evaluation**: Measure counterfactual quality with proximity, sparsity, and validity metrics
- **Production Ready**: Clean codebase with proper error handling and validation

## Project Structure

```
Counterfactual/
├── src/
│   ├── models/
│   │   └── cnn_classifier.py        # CNN for cats vs dogs classification
│   ├── counterfactuals/
│   │   └── gradient_based.py        # Gradient-based counterfactual generation
│   ├── utils/
│   │   ├── data_loader.py           # Real image data loader (Kaggle dataset)
│   │   ├── visualization.py         # Counterfactual visualization
│   │   └── metrics.py               # Evaluation metrics
│   ├── config.py                    # Configuration settings
│   └── __init__.py                  # Package initialization
├── data/
│   └── kaggle_cats_dogs/            # Real images (download from Kaggle)
│       ├── Cat/                     # Cat images
│       └── Dog/                     # Dog images
├── models/                          # Saved trained models
├── results/                         # Generated counterfactual results
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
1. Go to [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
2. Download and extract the dataset
3. Place folders in `data/kaggle_cats_dogs/`:
   ```
   data/kaggle_cats_dogs/
   ├── Cat/
   │   ├── 1.jpg
   │   ├── 2.jpg
   │   └── ...
   └── Dog/
       ├── 1.jpg
       ├── 2.jpg
       └── ...
   ```

## Quick Start

```python
from src.utils.data_loader import DataLoader
from src.models.cnn_classifier import CNNClassifier
from src.counterfactuals.gradient_based import GradientBasedCounterfactuals

# Load real Kaggle cats vs dogs dataset
loader = DataLoader()
images, labels = loader.load_kaggle_cats_dogs(
    data_path='data/kaggle_cats_dogs',
    max_samples_per_class=500
)

# Train CNN on real images
classifier = CNNClassifier(input_shape=(64, 64, 3))
model = classifier.build_model()
classifier.train(images, labels, epochs=20)

# Generate counterfactuals on real cat/dog photos
cf_generator = GradientBasedCounterfactuals(model)
result = cf_generator.generate_counterfactual(
    images[0],
    target_class=1,  # Flip to opposite class
    max_iterations=1000
)

# Visualize the counterfactual
from src.utils.visualization import CounterfactualVisualizer
viz = CounterfactualVisualizer()
viz.plot_counterfactual_comparison(
    images[0], 
    result['counterfactual'],
    result['original_pred'],
    result['cf_pred']
)
```

## Dataset Requirements

**Important**: This project requires real images. No synthetic fallback is provided.

### Kaggle Cats vs Dogs Dataset
- **Source**: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
- **Size**: ~800MB (thousands of cat and dog images)
- **Format**: JPG images in `Cat/` and `Dog/` folders
- **Required**: The application will raise an error if the dataset is not found

## How It Works

### Gradient-Based Counterfactual Generation
1. **Start with original image** (e.g., classified as "dog")
2. **Define target class** (e.g., "cat")
3. **Use model gradients** to find minimal pixel changes
4. **Optimize with dual objectives**:
   - Proximity loss: Keep changes minimal (L2 distance)
   - Sparsity loss: Change as few pixels as possible (L1 distance)
5. **Generate counterfactual** showing exactly what changed

### Algorithm
```
For each iteration:
  1. Compute model prediction gradient
  2. Update perturbation to move toward target class
  3. Apply proximity and sparsity constraints
  4. Check if target class is achieved
  5. Return counterfactual when successful
```

## Evaluation Metrics

- **Validity**: Does the counterfactual achieve the target class?
- **Proximity**: L2 distance from original image
- **Sparsity**: L1 distance (number of pixels changed)
- **Success Rate**: Percentage of successful counterfactual generations

## Why This Matters

### Explainable AI in Action
- **Transparency**: See exactly what the model "looks for" in real images
- **Debugging**: Understand model decision boundaries
- **Trust**: Verify that models make sensible distinctions between cats and dogs
- **Education**: Visual learning about deep learning internals

## Core Components

### 1. CNNClassifier (`src/models/cnn_classifier.py`)
- Binary image classifier for cats vs dogs
- Multi-layer CNN architecture with batch normalization
- Dropout for regularization
- Model save/load functionality

### 2. GradientBasedCounterfactuals (`src/counterfactuals/gradient_based.py`)
- Gradient-based optimization for counterfactual generation
- Proximity and sparsity loss functions
- Batch processing support
- Comprehensive evaluation metrics

### 3. DataLoader (`src/utils/data_loader.py`)
- Loads real Kaggle cats vs dogs dataset
- Image preprocessing and normalization
- Dataset shuffling and validation

### 4. Visualization & Metrics (`src/utils/`)
- Counterfactual visualization tools
- Quality metrics computation
- Side-by-side comparisons

---

**Built with**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib