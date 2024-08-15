# Example usage
# if __name__ == "__main__":
    # Define directories
    taxa_name = 'baetis_niger'
    label_dir = f'data/labels/{taxa_name}/obj_Train_data/'  # Directory containing YOLO v1.1 label files
    image_dir = f'data/images/{taxa_name}/'  # Directory containing the corresponding images

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    # Train the model with validation data
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_image_paths) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_image_paths) // batch_size,
        callbacks=[early_stopping]
    )

    with open(f'data/models/{taxa_name}_training_history.json', 'w') as f:
        json.dump(history.history, f)

    # Save the model
    # model.save("data/models/baetis_niger_bounding_box_model.h5")
    
    model.save(f"data/models/{taxa_name}_bounding_box_model.keras")

    print(f"Model training complete and saved as {taxa_name}_bounding_box_model.keras")
    
## evaluate the model 
# Example of loading the history
    with open(f'data/models/{taxa_name}_training_history.json', 'r') as f:
      history_data = json.load(f)
      

plt.plot(history_data['loss'], label = 'Training Loss')
plt.plot(history_data['val_loss'], label='Validation Loss')
plt.legend()
plt.show() 

 # Assuming you have defined a data_generator function as shown earlier
    steps_per_epoch = len(train_image_paths) // batch_size
    validation_steps = len(val_image_paths) // batch_size
# Load the saved .keras model
    model = tf.keras.models.load_model(f'data/models/{taxa_name}_bounding_box_model.keras')

    # Call the evaluate and plot function
    evaluate_plot_model(model, val_generator, steps_per_epoch, validation_steps)




