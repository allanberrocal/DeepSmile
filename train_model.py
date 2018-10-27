import create_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

IMAGE_STD = 255.0
TEST = False
test_ratio = 0.1
num_epochs = 5
batch_size = 100

model_path = "model.h5"

# load images
data_not_smiling = np.load('data_not_smiling_session.npy')
data_smiling = np.load('data_smiling_session.npy')
l_not_smiling = len(data_not_smiling)
l_smiling = len(data_smiling)
l_total = l_smiling + l_not_smiling
data = np.vstack((data_not_smiling, data_smiling))

# Plot some of the images
#plt.figure()
#plt.imshow(data_smiling[700])
#plt.show()

# Normalize the data
data = data / IMAGE_STD

# Expand the dimensions, tensorflow expects a colour channel
data = np.expand_dims(data, axis=3)

# Create labels
labels = np.zeros(l_total)
labels[l_smiling:] = 1

# create model
model = create_model.create_model2()

if TEST:
    # Split the data into train and test set; the data is shuffled before splitting
    samples_train, samples_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_ratio)

    # Train with validation data
    model.fit(samples_train, labels_train, validation_data=(samples_test, labels_test), epochs=num_epochs,
              batch_size=batch_size)
else:
    # Use all the data for training and store the final model
    model.fit(data, labels, epochs=num_epochs, batch_size=batch_size, shuffle=True)
    model.save(model_path)
