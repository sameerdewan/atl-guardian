# Imports
import os
import signal
import sys
from datetime import datetime

import cv2
import gspread
import pytz
import supervision as sv
from google.auth import default
from inference import InferencePipeline


def setup_google_sheets():
    creds, _ = default()
    googlesheets = gspread.authorize(creds)
    document = googlesheets.open_by_key('YOUR_GOOGLE_SHEET_KEY')
    worksheet = document.worksheet('SingleCameraTest')
    return worksheet


def setup_video_writer():
    video_info = (352, 240, 60)  # The size of the stream
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("nyc_traffic_timelapse.mp4",
                             fourcc, video_info[2], video_info[:2])
    return writer

# Interrupt Handling


def setup_signal_handler(writer):
    def signal_handler(sig, frame):
        writer.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def on_prediction(predictions, video_frame, writer, worksheet):
    # Process Results
    detections = sv.Detections.from_inference(predictions)
    annotated_frame = sv.BoundingBoxAnnotator(
        thickness=1
    ).annotate(video_frame.image, detections)

    # Add Frame To Timelapse
    writer.write(annotated_frame)

    # Format data for Google Sheets
    ET = pytz.timezone('America/New_York')
    time = datetime.now(ET).strftime("%H:%M")
    fields = [time, len(detections)]
    print(fields)

    # Add to Google Sheet
    worksheet.append_rows([fields], "USER_ENTERED")


def run_pipeline(api_key, worksheet, writer):
    pipeline = InferencePipeline.init(
        model_id="vehicle-detection-3mmwj/1",
        max_fps=0.5,
        confidence=0.3,
        video_reference="https://webcams.nyctmc.org/api/cameras/04e09ed5-2d97-4e29-8438-b87748850dbb/image",
        on_prediction=lambda predictions, video_frame: on_prediction(
            predictions, video_frame, writer, worksheet),
        api_key=api_key
    )

    pipeline.start()
    pipeline.join()


if __name__ == "__main__":
    api_key = os.getenv("ROBOFLOW_API_KEY")

    worksheet = setup_google_sheets()
    writer = setup_video_writer()

    setup_signal_handler(writer)

    run_pipeline(api_key, worksheet, writer)
