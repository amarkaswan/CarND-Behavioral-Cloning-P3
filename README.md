# Behavioral Cloning Project
#### This project intends to implement a software pipeline to clone human driving behavior using a convolutional neural network (CNN) in Keras. In particular, the software pipeline predicts steering angles for road images captured by a front-facing camera mounted on an autonomous vehicle. 

## Key Steps
This section briefly outlines the key steps employed to implement the desired software pipeline. 
* To begin with, I have utilized the training mode of [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) to collect the image data with their corresponding steering angles. Then, I have divided the whole dataset into three parts: training, validation, and testing sets.  
* Next, I have implemented NVIDIA's end-to-end deep learning approach, called [PilotNet](https://arxiv.org/pdf/1704.07911.pdf), to learn the relationship between image features and steering angles. 
* Finally, I have trained and validated the model using the training and validation sets, respectively. Then, I have tested the model using the testing set to check if the learned models perform equally well for the unseen data as it does for the training and validation sets. Besides, I have tested the model using the simulator in the autonomous mode to verify if the vehicle stays in the drivable area of the tracks.

Now, I will render a detailed discussion on each of the above-stated steps. 

### Step 1: Data Collection
<strong> Udacity's Self-Driving Car Simulator:</strong> This simulator is developed by [Udacity](https://www.udacity.com/) to teach students how to train autonomous vehicles for driving on the roads using deep learning. The autonomous vehicle in the simulator is equipped with three front-facing cameras, namely left, center and right. Below is a pictorial of the same.
<p></p>
<table>
 <center>
   <tr>
     <td> <img src="./examples/autonomous-vehicle.png" width="200" height="125"> </td>
  </tr>
 </center>
 </table>
 <p></p>

The simulator has two modes. The former mode is called a training mode through which we can collect images captured from three cameras with steering, throttle, brake, and speed measurements. In contrast, the latter mode is called an autonomous mode through which we can test the trained model, i.e., we can check if the model is able to keep the vehicle in the drivable portion of the road. Besides, the simulator has two tracks. The first track includes entries, exits, missing lane lines, bridges, and sharp turns, while the second track consists of uphill, downhill, bumps, steep and u-shaped turns.


Although the throttle, brake, and speed measurements could be useful to train a deep neural network, but in this project, I have used only the steering measurements. In particular, I have used images as the feature set and the steering measurements as the label set.


The value of steering measurements lies between -1 and 1. Herein, the negative and positive steer measurements determine if the vehicle is off to the left or right side of the road, respectively. On the other hand, the 0 steer measurement indicates that the vehicle is driving on the center of the road. However, it is worthwhile to mention that the recorded steering measurements correspond to the center camera, i.e., we do not have the steering measurements corresponding to the left and right cameras. Then, how can we use these images? The answer is that we can approximate their steer measurements and use these images as if they are coming from the center camera. Examples of images captured from left, center, right cameras are shown below.

<p></p>
<table>
 <center>
   <tr>
     <td>Image from Left Camera </td>
    <td> Image from Center Camera </td>
    <td> Image from Right Camera </td>
  </tr>
   <tr>
     <td> <img src="./examples/left.jpg" width="200" height="125"> </td>
    <td> <img src="./examples/center.jpg" width="200" height="125"> </td>
    <td> <img src="./examples/right.jpg" width="200" height="125"> </td>
  </tr>
 </center>
 </table>
 <p></p>

It can be observed that the image captured from the left camera and the right camera resembles as if we are driving on the left side and the right side of the road, respectively. In such situations, I would like to train the model to steer the vehicle a litter harder to the right whenever it sees an image similar to the left camera image. It is realized by adding a correction coefficient to the steer measurement recorded from the center camera. Likewise, the model should drive a little harder to the left for the images comparable to the right camera image. It is achieved by subtracting the correction coefficient from the recorded measurement. Hence, these images can teach the model how to recover (i.e., come back to the center of the road) when the vehicle goes off to the left or right side of the road.


Although Udacity has provided some sample driving data, but it was not sufficient to train a successful model. Therefore, I have collected more data using the simulator in the training mode. Particularly, I have followed the following strategy to collect more data.

* Two laps of clockwise driving around the first track.
* Two laps of anti-clockwise driving around the first track.
* One lap of clockwise driving around the second track.
* One lap of anti-clockwise driving around the second track.
* Some data for recovery from left and right side of the road from both the tracks.

The final [dataset](https://drive.google.com/file/d/1CdocKizqnD2FLT4QG6KNuuypv1DiBt_3/view?usp=sharing) has `130671` images along with their steering measurements. Then, I have shuffled and divided it into three parts: training (60%), validation (20%), and testing (20%) sets for reducing the overfitting. 

### Step 2: Model Architecture

Initially, I have implemented a modified version of [LeNet](https://en.wikipedia.org/wiki/LeNet) convolutional neural network architecture to develop the intended software pipeline. The code for the LeNet based software pipeline can be found in `model_LeNet` directroy. However, I did not obtain a successful model that could keep the vehicle within the drivable portion of the road.

Next, I have replicated NVIDIA's PilotNet architecture, which has been shown to work effectively in an actual autonomous vehicle. The `model.py` contains the code of implemented software pipeline, and the below figure summarizes the said architecture.
<p></p>
<table>
 <center>
   <tr>
     <td> <img src="./model_architecture.png" width="450" height="520"> </td>
  </tr>
 </center>
 </table>
 <p></p>
 
The first two layers are used to preprocess the input images. In particular, the first layer crops out the top `60 pixels` and bottom `20 pixels` of each image that include unnecessary details such as sky, trees, and hills, and hood of the car, respectively. The second layer first normalizes the pixel intensities of each image and then standardizes them using mean shifting. 

The subsequent five layers have consisted of convolutional layers followed by a flattened layer. Then, the last four layers are comprised of fully connected dense layers. Herein, I have applied the ReLu activation function at each convolution layer and dense intermediate layers for adding nonlinearity. Lastly, I have used a tangent hyperbolic activation function at the output layer since predicted steering measurement must lie in the range [-1, 1]. The model has `770619` number trainable parameters. 

### Step 3, Training, Validation and Testing
