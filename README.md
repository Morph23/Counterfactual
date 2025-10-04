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
- **Gradient-based Counterfactuals**: Find minimal pixel perturbations that flip predictions
- **Comprehensive Evaluation**: Measure counterfactual quality with proximity, sparsity, and validity metrics

## How To Use

1) Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies

```powershell
pip install -r requirements.txt
```

3) Download and prepare the dataset

1. Go to: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
2. Download and extract the dataset.
3. Ensure the folder structure matches:

```
data/kaggle_cats_dogs/
├── Cat/
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 1.jpg
    └── ...
```

4) Train the model (short demo) or load an existing model

Notes:
- `main.py` runs a short demo by default (trains on a reduced subset for speed). Edit `main.py` to change training epochs, batch size, or which images are used for demonstration.
- Training on a full dataset or running longer experiments requires more time and memory; consider running on a GPU-enabled machine.

5) View generated visualizations

After `main.py` finishes, example visualizations are saved in `results/`:

- `results/counterfactual_example.png` — single example (Original → CF → Heatmap)
- `results/counterfactual_batch.png` — multiple examples in a grid

6) Tips and customization

- To increase counterfactual quality, adjust `GradientBasedCounterfactuals` parameters (e.g., `max_iterations`, loss weights) in `src/counterfactuals/gradient_based.py` or via the `main.py` call.
- To run on more images, change the `test_images` slice in `main.py`.
- To save additional outputs, extend `src/utils/visualization.py` and call the desired plotting function in `main.py`.

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

**Built with**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib