import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO


# Function to load an image from a file and convert it to OpenCV format
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)


# Function to detect faces in an image using OpenCV
def detect_faces(image, scaleFactor, minNeighbors):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))
    return faces


# Function to get a link to download an image
def get_image_download_link(img, filename, text):
    """
    Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Streamlit app
st.title("Face Recognition App")
st.write("""
    **Instructions:**
    1. Upload an image using the "Upload Image" button.
    2. Adjust the `scaleFactor` and `minNeighbors` parameters to improve face detection.
    3. Choose the color of the rectangles drawn around the detected faces.
    4. Click the "Submit" button to process the image and display the result.
    5. Use the "Download Image" button to save the processed image with detected faces to your device.
""")

# File uploader
image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

# Parameters for face detection
scaleFactor = st.slider("Scale Factor", 1.01, 1.50, 1.10)
minNeighbors = st.slider("Min Neighbors", 1, 10, 5)
rectangle_color = st.color_picker("Choose Rectangle Color", "#FF0000")

if image_file is not None:
    # Load the image
    image = load_image(image_file)

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = detect_faces(image_rgb, scaleFactor, minNeighbors)

    # Convert the color from hex to BGR
    rectangle_color_bgr = tuple(int(rectangle_color.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), rectangle_color_bgr, 2)

    # Display the processed image
    st.image(image_rgb, caption='Processed Image', use_column_width=True)
    st.write(f"Detected {len(faces)} face(s) in the image.")

    # Provide a button to download the processed image
    result = Image.fromarray(image_rgb)
    st.markdown(get_image_download_link(result, "processed_image.png", "Download Image"), unsafe_allow_html=True)
