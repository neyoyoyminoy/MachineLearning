#Brendan Nellis     ujf306
#Intro to Machine Learning      Fall 2025
#Homework 1

#make sure to have tensorflow installed
#keras is like a top layer of tensorflow
#mnist is the handwritten dataset
from tensorflow.keras.datasets import mnist #given from Dr. Markopoulos

(trainX, trainY), (testX, testY) = mnist.load_data() #this will load data into the variables

print("Xtrain: " + str(trainX.shape)) #prints the trained vector
print("Ytrain: " + str(trainY.shape))

print("Xtest: " + str(testX.shape)) #prints the tested vector
print("Ytest: " + str(testY.shape))

