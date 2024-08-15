import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json

def evaluate_plot_model(model, val_generator, steps_per_epoch, validation_steps):
    """
    Evaluates and plots the performance of a Keras model using data generators.
    
    Parameters:
    - model: A trained Keras model.
    - val_generator: A generator function that yields batches of validation data.
    - steps_per_epoch: The number of steps (batches) to draw from the generator for evaluation.
    - validation_steps: The number of steps (batches) for validation (if any).
    
    Returns:
    - A summary of the model's performance including loss and metrics, and plots for loss and metrics (if available).
    """

    # Evaluate the model using the validation data generator
    evaluation = model.evaluate(val_generator, steps=validation_steps, verbose=1)
    print(f"Validation Loss: {evaluation[0]}")
    print(f"Validation MAE: {evaluation[1]}")

    # # Check if training history is available
    # if hasattr(model, 'history') and model.history is not None:
    #     history = model.history.history  # Access the stored history if present
    # 
    #     # Print history for debugging
    #     print("History keys:", history.keys())
    #     print("History contents:", history)
    # 
    #     plt.figure(figsize=(16, 8))
    # 
    #     # Plot Loss
    #     plt.subplot(1, 2, 1)
    #     if 'loss' in history:
    #         plt.plot(history['loss'], label='Training Loss')
    #     if 'val_loss' in history:
    #         plt.plot(history['val_loss'], label='Validation Loss')
    #     plt.title('Loss Over Epochs')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    # 
    #     # Plot MAE (or other metrics if different)
    #     plt.subplot(1, 2, 2)
    #     if 'mae' in history:
    #         plt.plot(history['mae'], label='Training MAE')
    #         if 'val_mae' in history:
    #             plt.plot(history['val_mae'], label='Validation MAE')
    #         plt.title('Mean Absolute Error Over Epochs')
    #     else:
    #         available_metrics = [key for key in history.keys() if key not in ['loss', 'val_loss']]
    #         if available_metrics:
    #             metric_name = available_metrics[0]  # Use the first available metric
    #             plt.plot(history[metric_name], label=f'Training {metric_name.capitalize()}')
    #             val_metric_name = f'val_{metric_name}'
    #             if val_metric_name in history:
    #                 plt.plot(history[val_metric_name], label=f'Validation {metric_name.capitalize()}')
    #             plt.title(f'{metric_name.capitalize()} Over Epochs')
    #         else:
    #             plt.title('No additional metrics available for plotting')
    # 
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Error')
    #     plt.legend()
    # 
    #     plt.tight_layout(pad=4.0)
    #     plt.show()
    else:
        print("No training history found, skipping plots.")
