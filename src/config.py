# Configuration settings for counterfactual explanations project

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Image classification settings
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
NUM_CLASSES = 2  # cats vs dogs
BATCH_SIZE = 32
EPOCHS = 20

# Counterfactual generation settings
MAX_ITERATIONS = 1000
LEARNING_RATE = 0.01
PROXIMITY_WEIGHT = 1.0
SPARSITY_WEIGHT = 0.1
DIVERSITY_WEIGHT = 0.5

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)