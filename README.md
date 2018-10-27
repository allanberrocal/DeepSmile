# DeepSmile
Recognizing if a computer user is smiling from webcam images employing a neural network

This project was created as part of a short hackathon at the [ACM SIGCHI Summer School on Computational Interaction 2018](https://www-edc.eng.cam.ac.uk/summerschool/) together with [Allan Berrocal](https://iss.unige.ch/staff/berrocal-allan/). 

The project contains four files. Their use is explained in the following. 

## image_recording.py
This script is used to collect training data. You have to run this twice, once while smiling, and once while not smiling (change the path at the end to store the data into different files). 
The face is automatically recognized and cropped and only this crop is stored (in greyscale). Per session, 1000 images are recored. While recording, you should vary your head pose a little to improve the robustness of the trained network. 
You can of course also record several sessions and later stack the data to combine them. 

## create_model.py
In this script the model is defined. For simplicity, Keras is used. Go ahead and define your own model if you want.

## train_model.py
This uses the recorded images to train a neural network (defined in create_model.py) to distinguish non-smiling faces from smiling faces. You can choose between two different modes: Setting `TEST = True`, then training will include validation, or setting `TEST = False`, then all the data will be used for training and the final model will be stored.
With a simple model using three convolutional layers one can already reach over 97% accuracy on validation data. 

## run_inference.py
This will load the model you trained previously, get the video stream from the webcam and classify the images from the stream, label them either "SMILING" or "NOT SMILING" and display the video stream with the labels.

