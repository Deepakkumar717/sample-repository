import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

# Check for GPU and set memory growth, if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"Error in GPU configuration: {e}")

# Path to your image dataset
data_dir = r"C:\Users\adeep\OneDrive\Documents\machine learning\Project work\data"

# Valid image extensions
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Cleaning up invalid images
for image_class in os.listdir(data_dir):
    image_class_path = os.path.join(data_dir, image_class)

    # Ensure the item is a directory before processing
    if os.path.isdir(image_class_path):
        for image in os.listdir(image_class_path):
            image_path = os.path.join(image_class_path, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f"Image not in ext list: {image_path}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image: {image_path}, error: {e}")
    else:
        print(f"Skipping non-directory file: {image_class_path}")

# Load dataset from directory with image size and batch settings
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(256, 256),  # Resize images to 256x256
    batch_size=32           # Set batch size
)

# Create an iterator to get batches
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Inspecting batch
print(f"Batch shape (images): {batch[0].shape}")  # Shape of image data
print(f"Batch shape (labels): {batch[1].shape}")  # Shape of labels

# Example: Access a single image
i = 0  # Index of the image
print(f"Image {i} shape: {batch[0][i].shape}")

# Get max value in the batch (check pixel values)
print(f"Max pixel value in batch: {np.max(batch[0])}")

# Scale the images between 0 and 1
scaled = batch[0] / 255
print(f"Max scaled pixel value: {scaled.max()}")

# Apply scaling to the entire dataset
data = data.map(lambda x, y: (x / 255, y))

# Create a new iterator after scaling
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# Display the max value of the scaled batch
print(f"Max scaled value in batch after mapping: {batch[0].max()}")

# Plotting the first 4 images from the batch
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(f"Label: {batch[1][idx]}")
plt.show()

# Dataset length
dataset_length = len(data)

# Split dataset sizes
train_size = int(dataset_length * 0.7)
val_size = int(dataset_length * 0.2)
test_size = int(dataset_length * 0.1)

# Print total sizes to verify
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# Create train, validation, and test datasets
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Optional: Print shapes of the split datasets
print(f"Train dataset size: {len(train)}")
print(f"Validation dataset size: {len(val)}")
print(f"Test dataset size: {len(test)}")
