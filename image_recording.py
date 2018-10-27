# This code is partly taken from https://github.com/ageitgey/face_recognition

import face_recognition
import cv2
import numpy as np


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

process_this_frame = True

num_images = 1000
width = 64
height = 64
# Setup the array to store the camera images
data = np.zeros((num_images, width, height))

# Setting first cropped frame
ret, cropped_frame = video_capture.read()
counter = 0
while counter < num_images:

    # Get a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Crop the face
        cropped_frame = frame[top:bottom, left:right]
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = cv2.resize(cropped_frame, (width, height))

        # Store cropped frame
        data[counter] = cropped_frame
        counter += 1

    # Display the resulting image
    cv2.imshow('Video', cropped_frame)
    #cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
#np.save('data_not_smiling_session.npy', data)
np.save('data_smiling_session.npy', data)
