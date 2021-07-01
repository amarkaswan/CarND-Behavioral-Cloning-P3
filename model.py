### Import used librarires
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
#from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.utils import shuffle

### Function to read the names of the images along with path of their directory
def load_image_names():
    samples = []
    #with open('/opt/data/driving_log_one_col.csv') as csvfile:
    with open('/opt/data/driving_log_one_col.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

### Function to slipt the names of images in the data set into testing, training and validation sets           
def test_train_validation_slip(samples):
    # Shuffle the images to avoid overfitting
    shuffle(samples)
    # Split data into two sets, one for training and validation (80 %) and one for testing (20%)
    train_and_val_samples, test_samples = train_test_split(samples, test_size = 0.2)
    # Shuffle the images to avoid overfitting
    shuffle(train_and_val_samples)
     # Split training (75% of first set) and validation data (25% of first set)
    train_samples, validation_samples = train_test_split(train_and_val_samples, test_size = 0.25)
    
    return train_samples, validation_samples, test_samples
    
### Generator function to read data in batches for training rather than storing all the training data in main memory
def generator(samples, batch_size = 256):
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
                if center_angle < -1:
                    center_angle = -1
                elif center_angle > 1:
                    center_angle = 1
                images.append(center_image)
                angles.append(center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
### Functin to define training, validation and testing generators
def define_generators(train_samples, validation_samples, test_samples):
    # Set our batch size
    batch_size = 256

    # define training, validation and testing generators
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples,
                                     batch_size=batch_size)
    test_generator = generator(test_samples, batch_size=batch_size)
    
    return train_generator, validation_generator, test_generator

### Function to define convolutional neural network (CNN) architecture 
def model_architecture(loss_function = 'mse', optimizer_name = 'adam'):
    # define a palin stack of layers using Sequential() on which we will add layers one by one using .add() function.
    model = Sequential()
    # add a cropping layer to remove the unwanted areas of the input image
    # input 160 x 320 x 3 ----> output 80 x 320 x 3
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    # add a Lambda layer to normalize the image and mean center it
    # input 80 x 320 x 3 ----> output 80 x 320 x 3
    model.add(Lambda(lambda x: (x / 255) - 0.5))
    # add a convolutional layer with 24 filters, each of 5 x 5, stride 2 x 2 and apply 'relu' activation with 'valid' (defualt) padding
    # input 80 x 320 x 3 ----> output 76 x 316 x 24
    model.add(Conv2D(filters = 24, kernel_size = 5, strides = 2, activation = 'relu'))
    # add a convolutional layer with 36 filters, each of 5 x 5, stride 2 x 2 and apply 'relu' activation with 'valid' (defualt) padding
    # input 38 x 158 x 24 ----> output 34 x 154 x 36
    model.add(Conv2D(filters = 36, kernel_size = 5, strides = 2, activation = 'relu'))
    # add a convolutional layer with 48 filters, each of 5 x 5, stride 2 x 2 and apply 'relu' activation with 'valid' (defualt) padding
    # input 17 x 77 x 36 ----> output 13 x 73 x 48
    model.add(Conv2D(filters = 48, kernel_size = 5, strides = 2, activation = 'relu'))
    # add a convolutional layer with 64 filters, each of 3 x 3 and apply 'relu' activation with 'valid' (defualt) padding
    # input 6 x 36 x 48 ----> output 4 x 34 x 64
    model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    # add a convolutional layer with 64 filters, each of 3 x 3 and apply 'relu' activation with 'valid' (defualt) padding
    # input 4 x 34 x 64 ----> output 2 x 32 x 64
    model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    # add a flat layer 
    # input 2 x 32 x 64 ----> output 4096
    model.add(Flatten())
    # add a dense layer and apply 'relu' activation
    # input 4096 ----> output 100
    model.add(Dense(100, activation = 'relu'))
    # add a dense layer and apply 'relu' activation
    # input 100---> output 50
    model.add(Dense(50, activation = 'relu'))
    # add a dense layer and apply 'relu' activation
    # input 50---> output 10
    model.add(Dense(10, activation = 'relu'))
    # add final dense layer and apply 'tanh' activation for getting output in the range [-1, 1]
    # input 10---> output 1
    model.add(Dense(1, activation = 'tanh'))

    # compile the model with loss function ('mse') and optimizer ('adm')
    model.compile(loss = loss_function, optimizer = optimizer_name)
    
    return model

### Function to train the model
def perform_training(model, train_generator, train_samples, validation_generator, validation_samples, batch_size = 256, epochs = 3, verbose = 1):
    # train the model using fit_generator() function 
    history = model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,
                   validation_steps = math.ceil(len(validation_samples)/batch_size), epochs= 5, verbose=1)
    # save the model
    model.save('model.h5')
    
    return history

### Function to test model using testing data
def perform_testing(model, test_generator, test_samples, batch_size = 256):
    # evaluate the model using testing data
    testing_mse = model.evaluate_generator(test_generator, steps = math.ceil(len(test_samples)/batch_size))

    return testing_mse
### Function to save training_mse, validation_mse and testing_mse   
def to_store(history, testing_mse):
    ### Save training, valdiation, and testing MSEs
    with open('train_val_mse.csv', 'w',newline = '') as f:
        write = csv.writer(f) 
        write.writerow(history.history['loss'])
        write.writerow(history.history['val_loss'])
        write.writerow([testing_mse])
        
def pipeline():
    samples = load_image_names()
    train_samples, validation_samples, test_samples = test_train_validation_slip(samples)
    generator(samples, batch_size = 256)
    train_generator, validation_generator, test_generator = define_generators(train_samples, validation_samples, test_samples)
    model = model_architecture(loss_function = 'mse', optimizer_name = 'adam')
    model.summary()
    history = perform_training(model, train_generator, train_samples, validation_generator, validation_samples)
    testing_mse = perform_testing(model, test_generator, test_samples)
    to_store(history, testing_mse)
    
pipeline()