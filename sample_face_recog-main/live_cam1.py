import face_recognition
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for a faster speed

    def load_encoding_images(self, images_path):
        """
        Load encoding images from folders in the given path
        :param images_path:
        :return:
        """
        # Iterate over each folder in the images_path
        for person_name in os.listdir(images_path):
            person_folder = os.path.join(images_path, person_name)
            if os.path.isdir(person_folder):
                images_path_list = glob.glob(os.path.join(person_folder, "*.png"))  # Load only PNG files
                print(f"{len(images_path_list)} encoding images found for {person_name}.")

                for img_path in images_path_list:
                    img = cv2.imread(img_path)

                    # Ensure the image is read correctly
                    if img is None:
                        print(f"Warning: {img_path} could not be read.")
                        continue

                    # Check the shape and dtype of the image
                    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

                    # Convert to RGB
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Check the shape and dtype after conversion
                    print(f"RGB Image shape: {rgb_img.shape}, dtype: {rgb_img.dtype}")

                    # Ensure the image is in 8-bit unsigned integer format
                    if rgb_img.dtype != np.uint8:
                        print(f"Converting image to uint8: {img_path}")
                        rgb_img = rgb_img.astype(np.uint8)

                    # Check if the image is in the expected range
                    if np.max(rgb_img) > 255 or np.min(rgb_img) < 0:
                        print(f"Warning: Image values out of range for {img_path}. Adjusting...")
                        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

                    # Show image for debugging
                    plt.imshow(rgb_img)
                    plt.title(f"Debug: {img_path}")
                    plt.axis("off")
                    plt.show()

                    # Detect faces in the image
                    try:
                        face_locations = face_recognition.face_locations(rgb_img)
                    except Exception as e:
                        print(f"Error while detecting faces in {img_path}: {e}")
                        continue

                    print(f"Face locations for {person_name}: {face_locations}")

                    # Check if any face is detected
                    if len(face_locations) > 0:
                        encodings = face_recognition.face_encodings(rgb_img, face_locations)
                        if len(encodings) > 0:
                            img_encoding = encodings[0]
                            # Store file name and file encoding
                            self.known_face_encodings.append(img_encoding)
                            self.known_face_names.append(person_name)
                            print(f"Face encoded for {person_name}")
                        else:
                            print(f"No encoding found for face in {person_name}")
                    else:
                        print(f"No face found in {person_name}")

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

def start_face_recognition():
    sfr = SimpleFacerec()
    sfr.load_encoding_images(r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\Face_samples")
    
    # Start video capture
    video_capture = cv2.VideoCapture(0)
    print("Starting the camera...")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Detect known faces in the frame
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw a label with a name below the face
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_face_recognition()
