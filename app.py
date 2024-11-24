import cv2
import numpy as np
import math
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image

# Initialize the HandDetector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

# Constants for image processing
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D"]

def process_frame(img):
    hands, img = detector.findHands(img)
    gesture = None
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size > 0:  # Check if imgCrop is not empty
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            gesture = labels[index]  # Capture the detected gesture label
    
    return gesture, img

st.title("Hand Gesture Recognition")
option = st.selectbox("Choose input type", ["Upload Video", "Live Video"])

if option == "Live Video":
    st.write("Click on 'Start Webcam' to use live video.")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        latest_gesture = st.empty()  # Placeholder to display the latest gesture
        video_placeholder = st.empty()  # Placeholder to display the video stream

        try:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break
                
                # Process each frame to detect hand gestures
                gesture, processed_img = process_frame(img)
                
                # Display the detected gesture label
                if gesture:
                    latest_gesture.write(f"Detected Gesture: {gesture}")
                
                # Convert BGR to RGB for Streamlit
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                video_placeholder.image(img_rgb, channels="RGB")
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()  # Ensure cap is released when exiting the loop
