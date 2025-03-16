import cv2
import streamlit as st
from ultralytics import YOLO
import torch
import yagmail
from collections import Counter
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time

# Initialize session state to store detected objects
if 'detected_objects' not in st.session_state:
    st.session_state['detected_objects'] = Counter()

# Function to reset detected objects
def reset_detected_objects():
    st.session_state['detected_objects'] = Counter()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load YOLOv8n model

# Email configuration (recommend using environment variables or Streamlit Secrets for passwords)
sender_email = st.secrets["email"]["sender_email"]
email_password = st.secrets["email"]["email_password"]
receiver_emails = st.text_area("Enter recipient email addresses (comma-separated)").split(',')
yag = yagmail.SMTP(user=sender_email, password=email_password)  # Replace with secure storage for password

# Function to send email notification
def send_email_notification(detected_object, subject="Object Detection Alert"):
    start_time = time.time()  # Start timing
    content = f"A {detected_object} was detected."
    yag.send(to=receiver_emails, subject=subject, contents=content)
    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate time taken
    st.success(f"Email notification sent for detected object: {detected_object} in {time_taken:.2f} seconds.")



# Function to detect objects in a video file
def detect_objects_in_video(video_path, confidence_threshold):
    cap = st.camera_input()
    detected_objects = Counter()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                if confidence >= confidence_threshold:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    detected_objects[label] += 1

    cap.release()
    return detected_objects

# Function to compare detected objects in two videos
def compare_objects(before_objects, after_objects):
    new_objects = {}
    for obj, count in after_objects.items():
        if obj not in before_objects or count > before_objects[obj]:
            new_objects[obj] = count
    return new_objects

# Function to export detected objects as CSV
def export_as_csv(detected_objects):
    csv_path = os.path.join(tempfile.gettempdir(), "detected_objects.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Object", "Count"])
        for obj, count in detected_objects.items():
            writer.writerow([obj, count])
    return csv_path

# Streamlit app
st.subheader("SCHOOL OF AERONAUTICAL ENGINEERING PRESENTS", divider='blue')
st.title(" Live video-based Object Detection")

# Get available cameras
available_cams = st.camera_input()
if available_cams:
    camera_index = st.selectbox("Select a Camera", available_cams)
else:
    st.warning("No camera found!")
    st.stop()

# Object selection for alerts
object_to_alert = st.multiselect("Select objects to receive alerts for", 
                                 list(model.names.values()))

# Confidence threshold slider
confidence_threshold = st.slider("Set Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5)

# State management for Start/Stop buttons
if 'is_detecting' not in st.session_state:
    st.session_state['is_detecting'] = False

start_detection = st.button("Start Live Detection")
stop_detection = st.button("Stop Live Detection")

if start_detection:
    st.session_state['is_detecting'] = True

if stop_detection:
    st.session_state['is_detecting'] = False

# Detection loop logic
if st.session_state['is_detecting']:
    cap = cv2.VideoCapture(camera_index)
    stframe = st.empty()  # Streamlit video stream display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video.")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)  # Perform inference

        # Extract predictions
        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if confidence >= confidence_threshold:
                    # Draw bounding boxes and labels on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Store detected objects in session state
                    st.session_state['detected_objects'][label] += 1

                    # Send email if the detected object is in the selected list
                    if label in object_to_alert:
                        send_email_notification(label)

        # Show the frame in Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()

# Report generation
# Report generation
if st.button("Generate Report") and st.session_state['detected_objects']:
    detected_objects = st.session_state['detected_objects']
    report = "Object Detection Report:\n"
    detected_object_types = set(detected_objects.keys())
    
    for obj in detected_object_types:
        report += f"{obj}\n"

    # Save the report to a file
    report_file_path = os.path.join(tempfile.gettempdir(), "detection_report.txt")
    with open(report_file_path, "w") as f:
        f.write(report)

    st.success("Report generated and saved as detection_report.txt.")

    # Download the report
    with open(report_file_path, "rb") as report_file:
        st.download_button("Download Report", report_file, file_name="detection_report.txt")


# Option to reset detected objects
if st.button("Reset Detected Objects"):
    reset_detected_objects()
    st.success("Detected objects have been reset.")

# Video comparison section
if st.checkbox("Compare Two Videos"):
    before_video_path = st.file_uploader("Upload Before Video", type=["mp4", "avi", "mov"])
    after_video_path = st.file_uploader("Upload After Video", type=["mp4", "avi", "mov"])

    if before_video_path and after_video_path:
        # Temporary files for uploaded videos
        before_video_temp = tempfile.NamedTemporaryFile(delete=False)
        before_video_temp.write(before_video_path.read())
        after_video_temp = tempfile.NamedTemporaryFile(delete=False)
        after_video_temp.write(after_video_path.read())

        # Detect objects in the before and after videos
        st.write("Detecting objects in 'Before' video...")
        before_objects = detect_objects_in_video(before_video_temp.name, confidence_threshold)
        st.write("Detecting objects in 'After' video...")
        after_objects = detect_objects_in_video(after_video_temp.name, confidence_threshold)

        # Compare the two sets of detected objects
        st.write("Comparing objects between 'Before' and 'After' videos...")
        new_objects = compare_objects(before_objects, after_objects)

        # Display the comparison result
        st.write("New objects detected or increased count in 'After' video:")
        for obj, count in new_objects.items():
            st.write(f"{obj}: {count}")

        # Plot new objects count
        if new_objects:
            fig, ax = plt.subplots()
            ax.bar(new_objects.keys(), new_objects.values())
            ax.set_title("New Objects Detected or Increased Count in 'After' Video")
            ax.set_xlabel("Objects")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Export the result as CSV
        csv_path = export_as_csv(new_objects)
        with open(csv_path, "rb") as f:
            st.download_button("Download Comparison CSV", f, file_name="new_objects_comparison.csv")
