import cv2
import numpy as np
import face_recognition

# Load the first image and get face encodings
img = cv2.imread(r"C:\Users\adeep\OneDrive\Documents\machine learning\Project work\images\download (1).jpeg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encodings = face_recognition.face_encodings(rgb_img)

# Check if a face was found in the first image
if len(img_encodings) > 0:
    img_encoding = img_encodings[0]  # Get the first face encoding
else:
    print("No face found in the first image.")
    exit()

# Load the second image and get face encodings
img2 = cv2.imread(r"C:\Users\adeep\OneDrive\Documents\machine learning\Project work\images\download.jpeg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encodings2 = face_recognition.face_encodings(rgb_img2)

# Check if a face was found in the second image
if len(img_encodings2) > 0:
    img_encoding2 = img_encodings2[0]  # Get the first face encoding
else:
    print("No face found in the second image.")
    exit()

# Compare faces
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result:", result)

# Display the images
cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
