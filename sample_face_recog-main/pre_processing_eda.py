import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

# Define the image folder path
image_folder_path = r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\pre_processing"

# Function for pre-processing
def pre_processing(image_folder_path):
    """
    Pre-process images: resize, convert to grayscale and normalize
    :param image_folder_path: Path to the folder containing images separated by person name
    :return: preprocessed_images, image_labels (list of images and corresponding labels)
    """
    preprocessed_images = []
    image_labels = []
    
    # Check if the base folder exists
    if not os.path.exists(image_folder_path):
        print(f"Error: The folder {image_folder_path} does not exist.")
        return preprocessed_images, image_labels
    
    # Loop over each person's folder
    for person_name in os.listdir(image_folder_path):
        person_folder = os.path.join(image_folder_path, person_name)
        
        # Check if the person's folder is a directory
        if not os.path.isdir(person_folder):
            print(f"Skipping {person_folder} as it is not a directory.")
            continue

        print(f"Processing folder: {person_folder}")
        
        # Loop through each image in the person's folder
        for img_file in glob.glob(os.path.join(person_folder, "*.jpg")):
            print(f"Found image: {img_file}")
            img = cv2.imread(img_file)
            
            if img is not None:
                # Convert to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize image to a standard size (e.g., 100x100)
                resized_img = cv2.resize(gray_img, (100, 100))
                
                # Normalize the image (values between 0 and 1)
                normalized_img = resized_img / 255.0
                
                preprocessed_images.append(normalized_img)
                image_labels.append(person_name)
            else:
                print(f"Warning: Could not read {img_file}")
    
    print(f"Preprocessed {len(preprocessed_images)} images.")
    return preprocessed_images, image_labels


# Function for EDA (Exploratory Data Analysis)
def eda(preprocessed_images, image_labels):
    """
    Perform exploratory data analysis on preprocessed images
    :param preprocessed_images: List of preprocessed images
    :param image_labels: List of corresponding labels (person names)
    """
    if len(preprocessed_images) == 0:
        print("No images to analyze.")
        return

    # Plot a sample of images
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(preprocessed_images[i], cmap='gray')
        ax.set_title(image_labels[i])
        ax.axis('off')
    plt.show()

    # Count the number of images per person
    label_counts = Counter(image_labels)
    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("Number of Images per Person")
    plt.xlabel("Person")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.show()

    # Use PCA to reduce the dimensionality of the images (flatten images first)
    flat_images = [img.flatten() for img in preprocessed_images]
    pca = PCA(n_components=2)
    image_pca = pca.fit_transform(flat_images)

    # Visualize the images in the reduced PCA space
    plt.figure(figsize=(10, 5))
    for person in set(image_labels):
        idxs = [i for i, label in enumerate(image_labels) if label == person]
        plt.scatter(image_pca[idxs, 0], image_pca[idxs, 1], label=person)
    plt.title("PCA of Images")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


# Execute pre-processing and EDA
preprocessed_images, image_labels = pre_processing(image_folder_path)
eda(preprocessed_images, image_labels)
