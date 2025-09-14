#Brendan Nellis     ujf306
#Intro to Machine Learning      Fall 2025
#Homework 1

#make sure to have tensorflow installed
#keras is like a top layer of tensorflow
#mnist is the handwritten dataset
from tensorflow.keras.datasets import mnist #given from Dr. Markopoulos

(train_X, train_y), (test_X, test_y) = mnist.load_data() #this will load data into the variables

print("X_train: " + str(train_X.shape)) #prints the trained vector
print("Y_train: " + str(train_y.shape))

print("X_test: " + str(test_X.shape)) #prints the tested vector
print("Y_test: " + str(test_y.shape))
