"""
Gradient-based coun    def generate_counterfactual(self,
                              original_image,
                              target_class=None,
                              proximity_weight=0.1,
                              sparsity_weight=0.001,
                              target_confidence=0.9,
                              learning_rate=0.1,
                              max_iterations=500,
                              verbose=False):l generation for image classification.
Implementation of gradient-based optimization to find minimal perturbations.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

class GradientBasedCounterfactuals:
    """Advanced gradient-based counterfactual generator for image classifiers."""
    
    def __init__(self, model):
        """Initialize the gradient-based counterfactual generator.
        
        Args:
            model: Trained CNN model (Keras model)
        """
        self.model = model
        self.model_type = 'image'
    
    def generate_counterfactual(self, 
                              original_image, 
                              target_class=None,
                              proximity_weight=1.0,
                              sparsity_weight=0.1,
                              target_confidence=0.9,
                              learning_rate=0.01,
                              max_iterations=500,
                              verbose=False):
        """Generate counterfactual explanation for an image.
        
        Args:
            original_image: Original image to explain (shape: H x W x C)
            target_class: Target class (None for opposite of current prediction)
            proximity_weight: Weight for L2 proximity loss
            sparsity_weight: Weight for sparsity loss (L1 norm)
            target_confidence: Target confidence for the counterfactual
            learning_rate: Learning rate for optimization
            max_iterations: Maximum optimization iterations
            verbose: Print optimization progress
            
        Returns:
            Dictionary containing counterfactual image and metadata
        """
        # Ensure image has batch dimension
        if len(original_image.shape) == 3:
            original_image = np.expand_dims(original_image, axis=0)
        
        # Get original prediction
        original_pred = self.model.predict(original_image, verbose=0)[0][0]
        original_class = int(original_pred > 0.5)
        original_confidence = original_pred if original_class == 1 else (1 - original_pred)
        
        # Set target class if not specified
        if target_class is None:
            target_class = 1 - original_class  # Flip for binary classification
        
        if verbose:
            print(f"Original class: {original_class} (confidence: {original_confidence:.4f})")
            print(f"Target class: {target_class}")
        
        # Initialize perturbation
        perturbation = tf.Variable(
            tf.zeros_like(original_image, dtype=tf.float32),
            trainable=True
        )
        
        # Convert to tensors
        original_tensor = tf.constant(original_image, dtype=tf.float32)
        target_tensor = tf.constant([[float(target_class)]], dtype=tf.float32)
        
        # Optimizer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        best_counterfactual = None
        best_loss = float('inf')
        
        for iteration in range(max_iterations):
            with tf.GradientTape() as tape:
                # Generate counterfactual
                counterfactual = original_tensor + perturbation
                counterfactual = tf.clip_by_value(counterfactual, 0.0, 1.0)
                
                # Get prediction
                prediction = self.model(counterfactual, training=False)
                
                # Classification loss (binary crossentropy)
                class_loss = tf.keras.losses.binary_crossentropy(target_tensor, prediction)
                
                # Proximity loss (L2 distance)
                proximity_loss = tf.reduce_mean(tf.square(perturbation)) * proximity_weight
                
                # Sparsity loss (L1 norm)
                sparsity_loss = tf.reduce_mean(tf.abs(perturbation)) * sparsity_weight
                
                # Total loss
                total_loss = class_loss + proximity_loss + sparsity_loss
            
            # Compute gradients and update
            gradients = tape.gradient(total_loss, [perturbation])
            optimizer.apply_gradients(zip(gradients, [perturbation]))
            
            # Check progress
            current_pred = prediction.numpy()[0][0]
            current_class = int(current_pred > 0.5)
            current_confidence = current_pred if current_class == 1 else (1 - current_pred)
            
            # Save best result
            if total_loss < best_loss:
                best_loss = total_loss
                best_counterfactual = counterfactual.numpy().copy()
            
            # Check if target achieved
            if current_class == target_class and current_confidence >= target_confidence:
                if verbose:
                    print(f"[SUCCESS] Target achieved at iteration {iteration+1}")
                break
            
            # Progress logging
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1}: loss={float(total_loss):.4f}, "
                      f"class={current_class}, confidence={current_confidence:.4f}")
        
        # Use best result if available
        if best_counterfactual is not None:
            counterfactual_image = best_counterfactual
        else:
            counterfactual_image = (original_tensor + perturbation).numpy()
            counterfactual_image = np.clip(counterfactual_image, 0, 1)
        
        # Final prediction
        final_pred = self.model.predict(counterfactual_image, verbose=0)[0][0]
        final_class = int(final_pred > 0.5)
        final_confidence = final_pred if final_class == 1 else (1 - final_pred)
        
        # Calculate metrics
        perturbation_norm = np.linalg.norm(perturbation.numpy())
        pixel_changes = np.sum(np.abs(perturbation.numpy()) > 0.01)
        total_pixels = np.prod(original_image.shape[1:])
        
        result = {
            'counterfactual': counterfactual_image[0],  # Remove batch dimension
            'original_image': original_image[0],
            'perturbation': perturbation.numpy()[0],
            'original_class': original_class,
            'original_pred': float(original_pred),
            'target_class': target_class,
            'final_class': final_class,
            'cf_class': final_class,  # Alias for compatibility
            'original_confidence': float(original_confidence),
            'final_confidence': float(final_confidence),
            'cf_confidence': float(final_confidence),  # Alias for compatibility
            'cf_pred': float(final_pred),  # Alias for compatibility
            'proximity': float(perturbation_norm),  # Alias for compatibility
            'sparsity': float(pixel_changes),  # Alias for compatibility
            'perturbation_norm': float(perturbation_norm),
            'pixel_changes': int(pixel_changes),
            'pixels_changed_percent': float(pixel_changes / total_pixels * 100),
            'success': final_class == target_class,
            'iterations': iteration + 1,
            'confidence_flip': abs(final_confidence - original_confidence)
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"COUNTERFACTUAL GENERATION COMPLETE")
            print(f"{'='*50}")
            print(f"Success: {'YES' if result['success'] else 'NO'}")
            print(f"Original: Class {original_class} (confidence: {original_confidence:.4f})")
            print(f"Final: Class {final_class} (confidence: {final_confidence:.4f})")
            print(f"Perturbation norm: {perturbation_norm:.6f}")
            print(f"Pixels changed: {pixel_changes:,} ({result['pixels_changed_percent']:.2f}%)")
            print(f"Iterations: {iteration + 1}")
        
        return result
    
    def generate_batch_counterfactuals(self, 
                                     images, 
                                     target_classes=None,
                                     verbose=False,
                                     **kwargs):
        """Generate counterfactuals for a batch of images.
        
        Args:
            images: Batch of images
            target_classes: Target classes for each image (optional)
            verbose: Print progress
            **kwargs: Additional arguments for generate_counterfactual
            
        Returns:
            List of counterfactual results
        """
        results = []
        n_images = len(images)
        
        if verbose:
            print(f"Generating counterfactuals for {n_images} images...")
        
        for i, image in enumerate(images):
            if verbose:
                print(f"\nProcessing image {i+1}/{n_images}")
            
            target = target_classes[i] if target_classes is not None else None
            
            result = self.generate_counterfactual(
                image, 
                target_class=target,
                verbose=verbose,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def evaluate_counterfactuals(self, results, verbose=True):
        """Evaluate quality of generated counterfactuals.
        
        Args:
            results: List of counterfactual results
            verbose: Print evaluation summary
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not results:
            return {}
        
        # Calculate aggregate metrics
        success_rate = np.mean([r['success'] for r in results])
        avg_perturbation_norm = np.mean([r['perturbation_norm'] for r in results])
        avg_pixel_changes = np.mean([r['pixel_changes'] for r in results])
        avg_pixels_changed_percent = np.mean([r['pixels_changed_percent'] for r in results])
        avg_iterations = np.mean([r['iterations'] for r in results])
        avg_confidence_flip = np.mean([r['confidence_flip'] for r in results])
        
        # Successful counterfactuals only
        successful = [r for r in results if r['success']]
        if successful:
            successful_perturbation_norm = np.mean([r['perturbation_norm'] for r in successful])
            successful_pixel_changes = np.mean([r['pixel_changes'] for r in successful])
            successful_pixels_changed_percent = np.mean([r['pixels_changed_percent'] for r in successful])
        else:
            successful_perturbation_norm = 0
            successful_pixel_changes = 0
            successful_pixels_changed_percent = 0
        
        evaluation_metrics = {
            'success_rate': success_rate,
            'avg_perturbation_norm': avg_perturbation_norm,
            'avg_pixel_changes': avg_pixel_changes,
            'avg_pixels_changed_percent': avg_pixels_changed_percent,
            'avg_iterations': avg_iterations,
            'avg_confidence_flip': avg_confidence_flip,
            'successful_perturbation_norm': successful_perturbation_norm,
            'successful_pixel_changes': successful_pixel_changes,
            'successful_pixels_changed_percent': successful_pixels_changed_percent,
            'total_attempts': len(results),
            'successful_attempts': len(successful)
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COUNTERFACTUAL EVALUATION RESULTS")
            print(f"{'='*60}")
            print(f"Success Rate: {success_rate:.1%} ({len(successful)}/{len(results)})")
            print(f"Average Perturbation Norm: {avg_perturbation_norm:.6f}")
            print(f"Average Pixels Changed: {avg_pixel_changes:.1f} ({avg_pixels_changed_percent:.2f}%)")
            print(f"Average Iterations: {avg_iterations:.1f}")
            print(f"Average Confidence Flip: {avg_confidence_flip:.4f}")
            
            if successful:
                print(f"\n🎯 SUCCESSFUL COUNTERFACTUALS ONLY:")
                print(f"Average Perturbation Norm: {successful_perturbation_norm:.6f}")
                print(f"Average Pixels Changed: {successful_pixel_changes:.1f} ({successful_pixels_changed_percent:.2f}%)")
        
        return evaluation_metrics

if __name__ == "__main__":
    print("Advanced Gradient-based Counterfactuals module ready!")
    print("Ready to generate high-quality counterfactual explanations!")
