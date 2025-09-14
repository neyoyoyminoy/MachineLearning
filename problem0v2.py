#Brendan Nellis     ujf306
#Intro to Machine Learning      Fall 2025
#Homework 1

#make sure to have tensorflow installed
#keras is like a top layer of tensorflow
#mnist is the handwritten dataset
import numpy as np
from tensorflow.keras.datasets import mnist #given from Dr. Markopoulos

(trainX, trainY), _ = mnist.load_data() #this will load MNIST data into the variables, leaving the 'test' set blank for simplicity

trainYmask = np.isin(trainY, [0,1,9]) #extracts only 0, 1, and 9 for analysis

print("Xtrain: " + str(trainX.shape)) #prints the trained vector
print("Ytrain: " + str(trainY.shape))

print("trainYmask: " + str(trainYmask.shape)) #prints the shape of the mask, this should be the same as the og array