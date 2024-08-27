import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import json
import io

# Define the neural network model
def create_model(input_shape):
    model = models.Sequential()
    
    # Define the input layer with the specified shape
    model.add(Input(shape=input_shape))

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    # Output layer: 4 units for the bounding box (x_min, y_min, x_max, y_max)
    model.add(layers.Dense(4, activation='linear'))

    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Data generator for images and bounding boxes
def data_generator(image_paths, bounding_boxes, batch_size, input_shape):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_images = []
            batch_bboxes = []

            for j in range(batch_size):
                if i + j >= len(image_paths):
                    break

                # Load and preprocess image
                image = Image.open(image_paths[i + j])
                
                # Check if the image is the correct size, resize if necessary
                if image.size != input_shape[:2]:
                    image = image.resize(input_shape[:2], Image.Resampling.LANCZOS)
                
                image = np.array(image) / 255.0
                batch_images.append(image)

                # Get corresponding bounding box
                batch_bboxes.append(bounding_boxes[i + j])

            yield np.array(batch_images), np.array(batch_bboxes)

# Function to load data from YOLO v1.1 format
def load_data_from_yolo(file_dir, image_dir):
    image_paths = []
    bounding_boxes = []

    for label_file in os.listdir(file_dir):
        if label_file.endswith('.txt'):
            # Assuming that the image file has the same name but with an image extension (e.g., .jpg)
            image_file = os.path.splitext(label_file)[0] + '.png'
            image_path = os.path.join(image_dir, image_file)

            with open(os.path.join(file_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    # Ensure that there are exactly 5 parts (class_id, x_center, y_center, width, height)
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

                        # Convert from YOLO format to bounding box format (x_min, y_min, x_max, y_max)
                        x_min = x_center - (width / 2)
                        y_min = y_center - (height / 2)
                        x_max = x_center + (width / 2)
                        y_max = y_center + (height / 2)

                        # Append image path and bounding box
                        image_paths.append(image_path)
                        bounding_boxes.append([x_min, y_min, x_max, y_max])
                    else:
                        print(f"Skipping malformed line in {label_file}: {line}")

    return image_paths, bounding_boxes

# Assuming you have a test dataset with images and corresponding bounding boxes
# Here, `test_image_paths` is a list of image file paths
# `test_bounding_boxes` is a list of ground truth bounding boxes for those images

# Function to load test data from YOLO v1.1 format
def load_test_data_from_yolo(file_dir, image_dir):
    image_paths = []
    bounding_boxes = []

    for label_file in os.listdir(file_dir):
        if label_file.endswith('.txt'):
            # Assuming that the image file has the same name but with an image extension (e.g., .jpg)
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_file)

            with open(os.path.join(file_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    # Ensure that there are exactly 5 parts (class_id, x_center, y_center, width, height)
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

                        # Convert from YOLO format to bounding box format (x_min, y_min, x_max, y_max)
                        x_min = x_center - (width / 2)
                        y_min = y_center - (height / 2)
                        x_max = x_center + (width / 2)
                        y_max = y_center + (height / 2)

                        # Append image path and bounding box
                        image_paths.append(image_path)
                        bounding_boxes.append([x_min, y_min, x_max, y_max])

    return image_paths, bounding_boxes

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, input_shape):
    image = Image.open(image_path)
    
    # Resize the image to the model's input shape
    if image.size != input_shape[:2]:
        image = image.resize(input_shape[:2], Image.Resampling.LANCZOS)
    
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Evaluate the model on the test set
def evaluate_model(model, test_image_paths, test_bounding_boxes, input_shape):
    predictions = []
    ground_truths = []
    
    for i, image_path in enumerate(test_image_paths):
        # Load and preprocess image
        image = load_and_preprocess_image(image_path, input_shape)
        
        # Predict bounding box
        predicted_bbox = model.predict(image)[0]
        predictions.append(predicted_bbox)
        ground_truths.append(test_bounding_boxes[i])
    
    # Convert lists to numpy arrays for easier calculation of metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Calculate Mean Absolute Error (MAE) as a simple evaluation metric
    mae = np.mean(np.abs(predictions - ground_truths), axis=0)
    print(f"Mean Absolute Error for bounding box coordinates: {mae}")
    
    # Calculate Intersection over Union (IoU) as a more relevant metric
    iou_scores = calculate_iou(predictions, ground_truths)
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou}")

def calculate_iou(pred_boxes, true_boxes):
    # Calculate IoU for each prediction-true box pair
    iou_scores = []
    for pred_box, true_box in zip(pred_boxes, true_boxes):
        xA = max(pred_box[0], true_box[0])
        yA = max(pred_box[1], true_box[1])
        xB = min(pred_box[2], true_box[2])
        yB = min(pred_box[3], true_box[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        boxBArea = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        iou_scores.append(iou)
    
    return np.array(iou_scores)
  
def create_model_with_classification(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    # Output layers:
    # 4 units for bounding box regression (x_min, y_min, x_max, y_max)
    bbox_output = layers.Dense(4, activation='linear', name='bbox_output')

    # num_classes units for classification (one-hot encoded)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')

    # Create the model with two outputs
    model = models.Model(inputs=model.input, outputs=[bbox_output, class_output])

    return model

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
  
def predict_bounding_boxes(model, image_paths, input_shape, batch_size=32):
    """Predict bounding boxes for a list of images using a trained model."""
    predictions = []
    
    #ensure batch_size in an integer
    batch_size = int(batch_size)

    # Loop over the image paths in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Preprocess the images in the current batch
        batch_images = np.vstack([load_and_preprocess_image(img_path, input_shape) for img_path in batch_paths])
        
        # Make predictions for the current batch
        batch_preds = model.predict(batch_images)
        
        # Store the predictions
        predictions.extend(batch_preds)

    return np.array(predictions)
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# 
# def evaluate_plot_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=10):
#     """
#     Trains, evaluates, and plots the performance of a Keras model.
#     
#     Parameters:
#     - model: A compiled Keras model.
#     - X_train: Training data (features).
#     - y_train: Training data (labels).
#     - X_test: Testing data (features).
#     - y_test: Testing data (labels).
#     - batch_size: Batch size for training.
#     - epochs: Number of epochs to train the model.
#     
#     Returns:
#     - A summary of the model's performance including loss and metrics, and plots for loss and metrics.
#     """
# 
#     # Train the model and store the training history
#     history = model.fit(X_train, y_train, 
#                         validation_data=(X_test, y_test), 
#                         epochs=epochs, 
#                         batch_size=batch_size,
#                         verbose=1)
#     
#     # Evaluate the model on the test set
#     test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Loss: {test_loss}")
#     print(f"Test MAE: {test_mae}")
#     
#     # Predict and calculate MAE manually for additional confirmation
#     y_pred = model.predict(X_test)
#     mae_manual = np.mean(np.abs(y_pred - y_test), axis=0)
#     print(f"Manual Mean Absolute Error (MAE): {mae_manual}")
#     
#     # Plot the training history
#     plt.figure(figsize=(12, 6))
# 
#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
# 
#     # Plot MAE (or other metrics if different)
#     plt.subplot(1, 2, 2)
#     if 'mae' in history.history:
#         plt.plot(history.history['mae'], label='Training MAE')
#         plt.plot(history.history['val_mae'], label='Validation MAE')
#         plt.title('Mean Absolute Error Over Epochs')
#     else:
#         # If using other metrics, adjust accordingly
#         metric_name = list(history.history.keys())[2]  # Get the first metric name after loss/val_loss
#         plt.plot(history.history[metric_name], label=f'Training {metric_name}')
#         plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name}')
#         plt.title(f'{metric_name.capitalize()} Over Epochs')
# 
#     plt.xlabel('Epochs')
#     plt.ylabel('Error')
#     plt.legend()
# 
#     plt.tight_layout()
#     plt.show()
# 
# 
#     evaluate_plot_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=10)
