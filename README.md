# Solution for Project 3 (Behavioral Cloning) of Udacity Self-Driving Car Nanodegree

This solution uses a neural network with only 15 parameters.

Initial idea based on: https://github.com/xslittlegrass/CarND-Behavioral-Cloning

Video of a lap of the track and the output of convolution and max pooling layer available here:
[![IMAGE ALT TEXT](http://img.youtube.com/vi/nkDNLf9ioRg/0.jpg)](https://youtu.be/nkDNLf9ioRg "CarND Behaviour cloning with just 15 params ")

## Getting started
1. Model. py doesn't work directly on the data from Udacity, but instead on data that has already been scaled and augmented. This speeds up training a lot.
2. That being said, the first step is to place the data in the udacity directory
3. Run prepare_data.py. It will go over all the images, scale them to 32x16, add augmented data and save them as numpy array in the files x.data.npy and y.data.npy
4. Run model.py. It will train the model using the data from x.data.npy and y.data.npy
5. Run drive.py model.json

## Creation of the training dataset 

I've used some of the ideas outlined here: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.w8yz2oycs

The dataset used for training is the one provided by Udacity with the following changes:

1. **Left/right camera** - Images from left/right camera are also used by modifiying the steering angle with 0.25 (6.25 degress). I've tried with different values and this works well. Using a constant is not exactly correct, because the correction should be dependent on the angle of the car, but this is not implemented
2. **Flipping** - Images with non-zero angle are flipped, including those from the left/right camera. This helps balance the neural network and not bias it to right or left turns. This is useful because the training data has more turns to the one side than to the other. There are a lot of zero angles in the dataset, so I don't flip them in order to balance them out.
3. **Brightness adjusment** - There is code for brightness adjusment, but is not currently used

## Model architecture

The final model consists of only 4 layers:

1. **Average Pooling with 2x2 pool size** - Further shrinks the images to 16x8
2. **Convolution with 2x2 kernel and 1x1 stride and ELU activation** - Scans the 16x8 picture with kernel size of 2x2 that is moved by 1 pixel at a time and produces 7x15x1 output. It helps with finding features in the input irrespective of their position in the input
3. **Max Pooling with 2x4 pool size** - Reduces the dimensions to 3x3x1.
4. **Dropout** - Prevents overfitting. This model has only 10 connections at this stage, but still using dropout improves performance 
5. **Dense layer with 1 neuron** - Used to sum up the data from the max pooling layer and output one variable - the steering angle
