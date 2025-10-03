"""
CNN Classifier for Image Counterfactual Explanations
Modern CNN architecture for binary cats vs dogs classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure TensorFlow uses CPU if GPU issues occur
try:
    # Try to configure GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    # Use CPU if GPU configuration fails
    tf.config.set_visible_devices([], 'GPU')

# Create directories
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)


class CNNClassifier:
    """CNN-based image classifier for binary classification."""
    
    def __init__(self, input_shape=(64, 64, 3), num_classes=2):
        """Initialize the CNN classifier.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes (2 for binary cats vs dogs)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.model_path = MODELS_DIR / 'cnn_classifier.h5'
    
    def build_model(self):
        """Build CNN architecture optimized for image counterfactuals."""
        
        model = keras.Sequential([
            # Input layer
            keras.Input(shape=self.input_shape),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block  
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling (better than flatten for counterfactuals)
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(1, activation='sigmoid') if self.num_classes == 2 else layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        if self.num_classes == 2:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        
        print("CNN Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, images, labels, validation_split=0.2, epochs=20, batch_size=32, verbose=1):
        """Train the CNN classifier.
        
        Args:
            images: Training images array
            labels: Training labels array
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
        """
        if self.model is None:
            self.build_model()
        
        print(f"Training CNN on {len(images)} images...")
        print(f"Image shape: {images.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            images, labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save model
        self.save_model()
        
        # Print final metrics
        val_loss, val_acc = self.model.evaluate(
            images[int(len(images)*validation_split):], 
            labels[int(len(labels)*validation_split):], 
            verbose=0
        )
        print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
        print(f"Final Validation Loss: {val_loss:.4f}")
        
        return self.history
    
    def predict(self, images):
        """Make predictions on images.
        
        Args:
            images: Images to predict on
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built or loaded!")
        
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        predictions = self.model.predict(images, verbose=0)
        
        if self.num_classes == 2:
            return predictions.flatten()  # Binary classification probabilities
        else:
            return predictions  # Multi-class probabilities
    
    def predict_class(self, images):
        """Predict class labels for images.
        
        Args:
            images: Images to predict on
            
        Returns:
            Predicted class labels
        """
        predictions = self.predict(images)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int)
        else:
            return np.argmax(predictions, axis=1)
    
    def evaluate(self, images, labels):
        """Evaluate model performance.
        
        Args:
            images: Test images
            labels: True labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded!")
        
        # Get predictions
        predictions = self.predict(images)
        predicted_classes = self.predict_class(images)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == labels)
        
        if self.num_classes == 2:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            precision = precision_score(labels, predicted_classes)
            recall = recall_score(labels, predicted_classes)
            f1 = f1_score(labels, predicted_classes)
            auc = roc_auc_score(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
        else:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(labels, predicted_classes, average='weighted')
            recall = recall_score(labels, predicted_classes, average='weighted')
            f1 = f1_score(labels, predicted_classes, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    
    def save_model(self, filepath=None):
        """Save the trained model.
        
        Args:
            filepath: Path to save model (optional)
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        if filepath is None:
            filepath = self.model_path
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model.
        
        Args:
            filepath: Path to load model from (optional)
        """
        if filepath is None:
            filepath = self.model_path
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        return self.model
    
    def get_layer_output(self, images, layer_name):
        """Get output from a specific layer (useful for counterfactuals).
        
        Args:
            images: Input images
            layer_name: Name of layer to extract
            
        Returns:
            Layer outputs
        """
        if self.model is None:
            raise ValueError("Model not built or loaded!")
        
        # Create intermediate model
        intermediate_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        return intermediate_model.predict(images, verbose=0)
    
    def plot_training_history(self):
        """Plot training history if available."""
        if self.history is None:
            print("No training history available!")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create classifier
    classifier = CNNClassifier(input_shape=(64, 64, 3))
    
    # Build model
    model = classifier.build_model()
    
    print("CNN Classifier ready for training!")
    print(f"Model will be saved to: {classifier.model_path}")