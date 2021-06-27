import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.utils import shuffle

samples = []
with open('/opt/data/driving_log_one_col.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
shuffle(samples)
train_n_val_samples, test_samples = train_test_split(samples, test_size = 0.2)
shuffle(train_n_val_samples)
train_samples, validation_samples = train_test_split(train_n_val_samples, test_size = 0.25)

def generator(samples, batch_size = 128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/opt/data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[1])
                # check if any angle is out of range [-1,1] and correct it
                """
                if center_angle > 1:
                    center_angle = 1
                elif center_angle < -1:
                    center_angle = -1
                 """
                images.append(center_image)
                angles.append(center_angle)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size = 256

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
test_generator = generator(test_samples, batch_size=batch_size)

#Callbacks for checkpoint and early stopping
#checkpoint = ModelCheckpoint(filepath = "model.h5", monitor='val_loss', save_best_only=True)
#stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience = 3)

## Model
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255) - 0.5))
model.add(Conv2D(24, 5, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(36, 5, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48, 5, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation = 'relu'))
model.add(Conv2D(64, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'tanh'))

model.compile(loss = 'mse', optimizer = 'adam')

history = model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,
                   validation_steps = math.ceil(len(validation_samples)/batch_size), epochs= 5, verbose=1)


#history = model.fit_generator(train_generator, steps_per_epoch = #math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps = math.ceil(len(validation_samples)/batch_size), #epochs= 3, verbose=1,  callbacks=[checkpoint, stopper])

model.save('model.h5')
### Evaluate model using test data and compute MSE
test_mse = model.evaluate_generator(test_generator, steps = math.ceil(len(test_samples)/batch_size))

### Save training, valdiation, and testing MSEs
with open('train_val_mse.csv', 'w',newline = '') as f:
    write = csv.writer(f) 
    write.writerow(history.history['loss'])
    write.writerow(history.history['val_loss'])
    write.writerow([test_mse])
