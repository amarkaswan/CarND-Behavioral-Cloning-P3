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

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Input, Dropout
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
                images.append(center_image)
                angles.append(center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
### Functin to define training, validation and testing generators
def define_generators(train_samples, validation_samples, test_samples):
    # Set our batch size
    batch_size = 128

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
    # add a convolutional layer with 6 filters, each of 5 x 5 and apply 'relu' activation with 'valid' (defualt) padding
    # input 80 x 320 x 3 ----> output 76 x 316 x 6
    model.add(Conv2D(6, 5, activation = 'relu'))
    # apply max polling with kernel_size 2 x 2 (default), stride 2 x 2 (default same as kernel_size)
    # input 76 x 316 x 6 ----> output 38 x 158 x 6
    model.add(MaxPooling2D())
    # add a dropout layer
    # input 38 x 158 x 6 ----> output 38 x 158 x 6
    model.add(Dropout(0.5))
    # add a convolutional layer with 6 filters, each of 5 x 5 and apply 'relu' activation with 'valid' (defualt) padding
    # input 38 x 158 x 24 ----> output 34 x 154 x 16
    model.add(Conv2D(16, 5, activation = 'relu'))
    # apply max polling with kernel_size 2 x 2 (default), stride 2 x 2 (default same as kernel_size)
    # input 34 x 154 x 16 ----> output 17 x 77 x 16
    model.add(MaxPooling2D())
    # add a dropout layer
    # input 17 x 77 x 16 ----> output 17 x 77 x 16
    model.add(Dropout(0.5))
    # add a flat layer 
    # input 38 x 158 x 16 ----> output 20944
    model.add(Flatten())
    # add a dense layer and apply 'relu' activation
    # input 20944 ----> output 120
    model.add(Dense(120, activation = 'relu'))
    # add a dense layer and apply 'relu' activation
    # input 120 ----> output 84
    model.add(Dense(84, activation = 'relu'))
    # add final dense layer and apply 'tanh' activation for getting output in the range [-1, 1]
    # input 84---> output 1
    model.add(Dense(1, activation = 'tanh'))

    # compile the model with loss function ('mse') and optimizer ('adm')
    model.compile(loss = loss_function, optimizer = optimizer_name)
    model.summary()
    
    return model

### Function to train the model
def perform_training(model, train_generator, train_samples, validation_generator, validation_samples, batch_size = 128, epochs = 5, verbose = 1):
    #Callbacks for checkpoint and early stopping
    checkpoint = ModelCheckpoint(filepath = "model.h5", monitor='val_loss', save_best_only=True)
    # train the model using fit_generator() function 
    history = model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(train_samples)/batch_size),validation_data=validation_generator, validation_steps = math.ceil(len(validation_samples)/batch_size), epochs = epochs, verbose = verbose, callbacks=[checkpoint])
    # save the model
    #model.save('model.h5')
    
    return history

### Function to test model using testing data
def perform_testing(model, test_generator, test_samples, batch_size = 128):
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
    generator(samples, batch_size = 128)
    train_generator, validation_generator, test_generator = define_generators(train_samples, validation_samples, test_samples)
    model = model_architecture(loss_function = 'mse', optimizer_name = 'adam')
    #model.summary()
    history = perform_training(model, train_generator, train_samples, validation_generator, validation_samples)
    testing_mse = perform_testing(model, test_generator, test_samples)
    to_store(history, testing_mse)
    
pipeline()
