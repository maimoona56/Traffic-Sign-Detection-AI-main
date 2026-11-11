import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚦 Traffic Sign Detection System")
st.write("Detect objects using YOLO from images, camera, videos and screen captures.")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)

st.sidebar.info(
    "Higher confidence = fewer but more confident detections."
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🖼️ Add Image",
        "📷 Camera",
        "🎥 Video",
        "🖥️ Screen Capture"
    ]
)

# ==================================================
# IMAGE TAB
# ==================================================

with tab1:

    st.header("🖼️ Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("🔍 Detect Objects", key="image_detect"):

            results = model(
                image,
                conf=confidence
            )

            result_image = results[0].plot()

            with col2:
                st.subheader("Detection Result")
                st.image(
                    result_image,
                    channels="BGR",
                    use_container_width=True
                )

            # Detection information
            boxes = results[0].boxes

            st.divider()
            st.subheader("📊 Detection Information")

            if boxes is not None and len(boxes) > 0:

                class_ids = boxes.cls.cpu().numpy().astype(int)

                names = results[0].names

                detected_names = [
                    names[class_id]
                    for class_id in class_ids
                ]

                st.success(
                    f"Detected Objects: {len(detected_names)}"
                )

                st.write(
                    "Detected Classes:",
                    ", ".join(detected_names)
                )

            else:
                st.warning("No objects detected.")

# ==================================================
# CAMERA TAB
# ==================================================

with tab2:

    st.header("📷 Laptop Camera")

    st.write(
        "Allow camera access when your browser asks for permission."
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Camera Image")
            st.image(image, use_container_width=True)

        if st.button("🔍 Detect From Camera", key="camera_detect"):

            results = model(
                image,
                conf=confidence
            )

            result_image = results[0].plot()

            with col2:
                st.subheader("Detection Result")
                st.image(
                    result_image,
                    channels="BGR",
                    use_container_width=True
                )

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                st.success(
                    f"Detected Objects: {len(boxes)}"
                )
            else:
                st.warning("No objects detected.")

# ==================================================
# VIDEO TAB
# ==================================================

with tab3:

    st.header("🎥 Video Detection")

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if video_file is not None:

        st.video(video_file)

        if st.button("▶️ Start Video Detection"):

            # Temporary input video
            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(video_file.read())
            input_file.close()

            # Open video
            cap = cv2.VideoCapture(input_file.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 20

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(0)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            frame_number = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(
                    frame,
                    conf=confidence,
                    verbose=False
                )

                annotated_frame = results[0].plot()

                out.write(annotated_frame)

                frame_number += 1

                if total_frames > 0:
                    progress.progress(
                        min(
                            frame_number / total_frames,
                            1.0
                        )
                    )

            cap.release()
            out.release()

            st.success("✅ Video detection completed!")

            with open(output_path, "rb") as video:

                st.download_button(
                    "⬇️ Download Detected Video",
                    data=video,
                    file_name="traffic_detection.mp4",
                    mime="video/mp4"
                )

            st.video(output_path)

            os.remove(input_file.name)

# ==================================================
# SCREEN CAPTURE TAB
# ==================================================

with tab4:

    st.header("🖥️ Screen Capture")

    st.write(
        "Take a screenshot of your current screen and run YOLO detection on it."
    )

    if st.button("📸 Capture Screen"):

        try:

            import pyautogui

            screenshot = pyautogui.screenshot()

            screenshot = screenshot.convert("RGB")

            st.session_state["screen_image"] = screenshot

        except Exception as e:

            st.error(
                f"Screen capture failed: {e}"
            )

    if "screen_image" in st.session_state:

        screen_image = st.session_state["screen_image"]

        st.subheader("Captured Screen")

        st.image(
            screen_image,
            use_container_width=True
        )

        if st.button(
            "🔍 Detect On Screen",
            key="screen_detect"
        ):

            results = model(
                screen_image,
                conf=confidence
            )

            result_image = results[0].plot()

            st.subheader("Detection Result")

            st.image(
                result_image,
                channels="BGR",
                use_container_width=True
            )

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:

                st.success(
                    f"Detected Objects: {len(boxes)}"
                )

            else:

                st.warning(
                    "No objects detected."
                )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🚦 Traffic Sign Detection | YOLO + Streamlit"
)