import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Prepare your dataset

# Step 2: Set up your project environment

# Step 3: Import the required libraries

# Step 4: Load and preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './data',
    target_size=(150, 150),
    batch_size=32)
print(train_generator.class_indices)
# Step 5: Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 6: Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(train_generator)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Step 8: Save the trained model
model.save('model.h5')

# # Step 9: Predict using the trained model
# loaded_model = tf.keras.models.load_model('path/to/saved/model')
# # Perform necessary preprocessing steps on new images before passing them through the model
# predictions = loaded_model.predict(new_images)
