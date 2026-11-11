import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display original image
    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # YOLO prediction
    results = model(image)

    # Draw detections
    annotated_image = image.copy()
    annotator = Annotator(annotated_image)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = f"{model.names[class_id]} {confidence:.2f}"

            annotator.box_label(
                box.xyxy[0],
                label
            )

    annotated_image = annotator.result()

    # Display result
    st.subheader("Detection Result")
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))