# This code is partly taken from https://github.com/ageitgey/face_recognition

import face_recognition
import cv2
import tensorflow as tf
import numpy as np

# Initialize some variables for recognition and load the smile recognition model
smiling_prob = 0.5
smiling_prob_threshold = 0.8
smoothing_factor = 0.75
IMAGE_STD = 255.0
width = 64
height = 64
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
font = cv2.FONT_HERSHEY_DUPLEX
face_locations = []
process_this_frame = True

while True:
    # Get a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # crop the face
        cropped_frame = frame[top:bottom, left:right]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Create crop as in image recording
        cropped_frame = frame[top:bottom, left:right]
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = cv2.resize(cropped_frame, (width, height))
        cropped_frame = cropped_frame / IMAGE_STD
        # Expanding the dimension so that the final shape is (1, 64, 64, 1) -> one image of width = 64, height = 64,
        # and with  1 channel
        cropped_frame = np.expand_dims(cropped_frame, axis=0)
        cropped_frame = np.expand_dims(cropped_frame, axis=3)

        # Get predictions
        probs = model.predict(cropped_frame)
        current_prob = probs[0][1]

        # Smooth the result
        smiling_prob = smoothing_factor * smiling_prob + (1 - smoothing_factor) * current_prob
        print("Current prob:" + str(current_prob) + ", Smiling prob: " + str(smiling_prob))
        if smiling_prob >= smiling_prob_threshold:
            text = "SMILING"
        else:
            text = "NOT SMILING"
        cv2.putText(frame, text, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
