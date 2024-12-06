import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image at {img_path}. Skipping.")
                continue

            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Check if the image is 8-bit
            if rgb_img.dtype == np.uint8 and rgb_img.ndim == 3 and rgb_img.shape[2] == 3:
                print(f"The image {img_path} is a valid 8-bit RGB image.")
            else:
                print(f"The image {img_path} is NOT an 8-bit RGB image.")
                continue  # Skip this image if it's not valid

            # Debugging: Print the shape and type of the image
            print(f"Image shape: {rgb_img.shape}, dtype: {rgb_img.dtype}")

            # Ensure image data type is correct for face_recognition
            if rgb_img.dtype != np.uint8:
                print(f"Error: Image {img_path} is not of type uint8. Current dtype: {rgb_img.dtype}")
                continue

            # Detect faces in the image
            try:
                face_locations = face_recognition.face_locations(rgb_img)
                print(f"Face locations for {os.path.basename(img_path)}: {face_locations}")

                if len(face_locations) > 0:
                    encodings = face_recognition.face_encodings(rgb_img, face_locations)
                    if len(encodings) > 0:
                        img_encoding = encodings[0]
                        self.known_face_encodings.append(img_encoding)
                        self.known_face_names.append(os.path.basename(img_path))
                        print(f"Face encoded for {os.path.basename(img_path)}")
                    else:
                        print(f"No encoding found for face in {os.path.basename(img_path)}")
                else:
                    print(f"No face found in {os.path.basename(img_path)}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
