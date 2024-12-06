import face_recognition
import cv2
import os
from tkinter import Tk, filedialog

# Function to load images from the dataset folder and get face encodings
def load_known_faces(dataset_folder):
    known_encodings = []
    known_names = []

    # Loop over the image files in the dataset
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure it's an image file
            image_path = os.path.join(dataset_folder, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Encode the face from the image (assuming each image has only one face)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:  # Proceed only if a face is found
                known_encodings.append(encodings[0])  # Add the encoding to the list
                known_names.append(os.path.splitext(filename)[0])  # Use the filename (without extension) as the name

    return known_encodings, known_names

# Function to process the target image and find matches
def recognize_faces_in_image(target_image_path, known_encodings, known_names):
    # Load the target image
    target_image = face_recognition.load_image_file(target_image_path)

    # Find all face locations and face encodings in the target image
    face_locations = face_recognition.face_locations(target_image)
    face_encodings = face_recognition.face_encodings(target_image, face_locations)

    # Convert the image to BGR color for OpenCV display (optional for showing the image)
    target_image_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

    # Loop over each face found in the target image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if this face matches any of the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(target_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name of the person
        cv2.putText(target_image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the result (optional)
    cv2.imshow("Image", target_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to allow users to select a file using a file dialog
def select_image_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    target_image_path = filedialog.askopenfilename(title="Select an Image", 
                                                   filetypes=[("Image files", "*.jpg *.png")])
    return target_image_path

# Main function
if __name__ == "__main__":
    # Specify the folder containing dataset images (known faces)
    dataset_folder = r"C:\Users\adeep\OneDrive\Documents\machine learning\Project work\images"

    # Load known faces and their encodings
    known_encodings, known_names = load_known_faces(dataset_folder)

    # Prompt the user to select the target image file via drag-and-drop or file dialog
    target_image_path = select_image_file()

    if target_image_path:  # Proceed only if an image was selected
        # Recognize and display the results
        recognize_faces_in_image(target_image_path, known_encodings, known_names)
    else:
        print("No image selected.")
