import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = input('Enter the name of image:')
img = cv2.imread(image_path+'.jpg')

# Check if the image was loaded successfully
if img is None:
    raise ValueError("Image not found at the specified path.")

# Convert the image to grayscale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces in the grayscale image
faces = face_classifier.detectMultiScale(
    grey, scaleFactor=1.1, minNeighbors=5
)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert BGR image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with rectangles
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
