import os
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Function to analyze image dimensions and face counts
def analyze_image_properties(dataset_folder):
    image_sizes = []
    face_counts = []
    known_names = []
    
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure it's an image file
            image_path = os.path.join(dataset_folder, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Get image dimensions
            h, w = image.shape[:2]
            image_sizes.append((w, h))

            # Detect faces in the image
            face_locations = face_recognition.face_locations(image)
            face_counts.append(len(face_locations))

            # Collect names (file names without extension)
            known_names.append(os.path.splitext(filename)[0])
    
    return image_sizes, face_counts, known_names

# Function to analyze face encoding distances between known faces
def analyze_face_encodings(known_encodings):
    if len(known_encodings) > 1:
        # Calculate distances between face encodings (pairwise)
        distances = face_recognition.face_distance(known_encodings, known_encodings[0])
        return distances
    return None

# Function to plot distribution of image sizes
def plot_image_size_distribution(image_sizes):
    widths, heights = zip(*image_sizes)
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, c='blue', label="Image Sizes (Width vs Height)")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Distribution of Image Sizes in Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot the number of faces per image
def plot_face_count_distribution(face_counts):
    plt.figure(figsize=(8, 6))
    plt.hist(face_counts, bins=np.arange(1, max(face_counts) + 2) - 0.5, rwidth=0.8, color='green')
    plt.xlabel("Number of Faces Detected")
    plt.ylabel("Number of Images")
    plt.title("Face Count Distribution in Dataset")
    plt.grid(True)
    plt.show()

# Function to plot class (person) distribution
def plot_class_distribution(known_names):
    name_counter = Counter(known_names)
    labels, values = zip(*name_counter.items())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='purple')
    plt.xlabel("Person")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution (Images per Person)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Function to plot face encoding distance distribution
def plot_face_encoding_distances(distances):
    if distances is not None:
        plt.figure(figsize=(8, 6))
        plt.hist(distances, bins=10, color='red', alpha=0.7)
        plt.xlabel("Face Encoding Distance")
        plt.ylabel("Frequency")
        plt.title("Distribution of Face Encoding Distances")
        plt.grid(True)
        plt.show()
    else:
        print("Not enough encodings to calculate distances.")

# Main function for EDA
if __name__ == "__main__":
    # Specify the dataset folder containing known images
    dataset_folder = r"C:\Users\adeep\OneDrive\Documents\machine learning\Project work\images"
    
    # Load known faces and their encodings
    known_encodings = []
    known_names = []

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure it's an image file
            image_path = os.path.join(dataset_folder, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Encode the face from the image (assuming each image has only one face)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:  # Proceed only if a face is found
                known_encodings.append(encodings[0])  # Add the encoding to the list
                known_names.append(os.path.splitext(filename)[0])  # Use the filename (without extension) as the name
    
    # Analyze image properties
    image_sizes, face_counts, known_names = analyze_image_properties(dataset_folder)
    
    # Plot the distribution of image sizes
    plot_image_size_distribution(image_sizes)
    
    # Plot the number of faces detected in each image
    plot_face_count_distribution(face_counts)
    
    # Plot the distribution of classes (number of images per person)
    plot_class_distribution(known_names)
    
    # Analyze and plot face encoding distances
    distances = analyze_face_encodings(known_encodings)
    plot_face_encoding_distances(distances)
