import time
import cv2
import numpy as np
import requests
from inference import get_model
import streamlit as st


# Configuration
FRAME_RATE = 1  # frames per second
VIDEO_SIZE = (700, 480)

# Initialize model
model = get_model(model_id="yolov8n-640")


def process_image(image_url):
    # Perform inference
    results = model.infer(image=image_url, confidence=0.1, iou_threshold=0.1)

    detections = results[0].predictions
    filter_classes = {
        "car": (255, 0, 0),
        "truck": (0, 255, 0),
        "bus": (0, 0, 255),
        "motorcycle": (255, 255, 0),
        "bicycle": (0, 255, 255),
        "person": (255, 0, 255),
    }

    # Fetch the image to annotate
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        response.raw.decode_content = True
        image = np.asarray(bytearray(response.raw.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        print(
            f"Failed to fetch image for annotation. Status code: {response.status_code}"
        )
        return None

    # Annotate frame
    for det in detections:
        if det.class_name in filter_classes:
            x1 = int(det.x - det.width / 2)
            y1 = int(det.y - det.height / 2)
            x2 = int(det.x + det.width / 2)
            y2 = int(det.y + det.height / 2)
            label = f"{det.class_name}: {det.confidence:.2f}"
            color = filter_classes[det.class_name]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    return image


def main(image_url):
    cam_footage = st.empty()

    while True:
        annotated_image = process_image(image_url)

        if annotated_image is not None:
            # Resize the image to match video size
            annotated_image = cv2.resize(annotated_image, VIDEO_SIZE)

            # Display the image
            cam_footage.image(annotated_image)

        # Wait for the next frame
        time.sleep(1 / FRAME_RATE)


if __name__ == "__main__":
    st.set_page_config(page_title="ATL Guardian", page_icon="🛡️", layout="centered")
    st.title("ATL Guardian")

    cam_select = st.selectbox(
        "Select a camera",
        [
            "Camera 1",
            "Camera 2",
        ],
    )

    cam_data = {
        "Camera 1": {
            "url": "https://webcams.nyctmc.org/api/cameras/04e09ed5-2d97-4e29-8438-b87748850dbb/image"
        },
        "Camera 2": {
            "url": "https://webcams.nyctmc.org/api/cameras/04e09ed5-2d97-4e29-8438-b87748850dbb/image"
        },
    }

    IMAGE_URL = cam_data[cam_select]["url"]

    main(image_url=IMAGE_URL)
