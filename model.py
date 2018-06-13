import os
import csv
import cv2
import numpy as np
import sklearn
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from random import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D

# Read Driving Log
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Tuning Parameters
camera_adjustment = [0.0, 0.2, -0.2]

# Split Data set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    name = './IMG/'+source_path.split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3]) + camera_adjustment[i]
                    images.append(center_image)
                    angles.append(center_angle)

                    # Add Flipped Image
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Model Architecture
model = Sequential()

# Normalization Layers
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Convolution Layer 1; 5x5 kernel
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))

# Convolution Layer 2; 5x5 kernel
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))

# Convolution Layer 3; 5x5 kernel
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

# Convolution Layer 4; 3x3 kernel
model.add(Convolution2D(64, 3, 3, activation='relu'))

# Convolution Layer 1; 3x3 kernel
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())

# Fully Connected Layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples) * 6,
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=3,
    verbose=1
)

model.save('model.h5')

exit()
