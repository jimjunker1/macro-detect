import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Define the neural network model
def create_model(input_shape):
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
                image = tf.keras.preprocessing.image.load_img(image_paths[i + j], target_size=input_shape[:2])
                image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
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
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_file)

            with open(os.path.join(file_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
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

# Example usage
if __name__ == "__main__":
    # Define directories
    label_dir = 'labels/'  # Directory containing YOLO v1.1 label files
    image_dir = 'images/'  # Directory containing the corresponding images

    # Load data from YOLO files
    image_paths, bounding_boxes = load_data_from_yolo(label_dir, image_dir)

    # Parameters
    input_shape = (224, 224, 3)  # Example input shape (height, width, channels)
    batch_size = 2
    epochs = 10
    test_size = 0.2  # 20% for validation

    # Split the data into training and validation sets
    train_image_paths, val_image_paths, train_bboxes, val_bboxes = train_test_split(
        image_paths, bounding_boxes, test_size=test_size, random_state=42
    )

    # Create and compile the model
    model = create_model(input_shape)
    model = compile_model(model)

    # Create data generators
    train_generator = data_generator(train_image_paths, train_bboxes, batch_size, input_shape)
    val_generator = data_generator(val_image_paths, val_bboxes, batch_size, input_shape)

    # Train the model with validation data
    model.fit(
        train_generator,
        steps_per_epoch=len(train_image_paths) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_image_paths) // batch_size
    )

    # Save the model
    model.save("bounding_box_model.h5")

    print("Model training complete and saved as bounding_box_model.h5")
