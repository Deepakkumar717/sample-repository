import cv2
from simple_facerec import SimpleFacerec

# Initialize the SimpleFacerec class
sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\adeep\OneDrive\Documents\machine learning\Project work\images")

# Try to open the webcam (or a video file)
cap = cv2.VideoCapture(0)  # Use the correct camera index

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect known faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        for face_loc, name in zip(face_locations, face_names):
            # Unpack the face location coordinates
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Corrected line to use name properly
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows

    cap.release()
    cv2.destroyAllWindows()
