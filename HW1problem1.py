#Brendan Nellis     ujf306
#Intro to Machine Learning      Fall 2025
#Homework 1
#Calculate the average vector (centroid) of each of the 3 digits in X.
#Compute: Compute the 1-norm, 2-norm, and 3-norm of each centroid vector.


import numpy as np
'''the next five lines are from problem 0'''
from tensorflow.keras.datasets import mnist #given from Dr. Markopoulos (problem 0)

(trainX, trainY), _ = mnist.load_data() #this will load MNIST data into the variables, leaving the 'test' set blank for simplicity
mask = np.isin(trainY, [0,1,9]) #extracts only 0, 1, and 9 for analysis
trainingImages = trainX[mask] #grabbing the masked images for vectorization
vTrainingImages = trainingImages.reshape(trainingImages.shape[0], -1) #flattens/reshapes (matrix vectorization) images of 0, 1, and 9; the '-1'

digit0 = vTrainingImages[trainY[mask] == 0] #seperated each training digit
digit1 = vTrainingImages[trainY[mask] == 1]
digit9 = vTrainingImages[trainY[mask] == 9]

centroid0 = np.mean(digit0, axis=0) #the 'mean' should find the centroid of the matrix
centroid1 = np.mean(digit1, axis=0)
centroid9 = np.mean(digit9, axis=0)

'''
print("centroid 0: ", centroid0.shape)
print("centroid 1: ", centroid1.shape)
print("centroid 9: ", centroid9.shape)
'''

manNorm0 = np.linalg.norm(centroid0, ord=1) #1-norm aka manhattan norm versions of each digit
manNorm1 = np.linalg.norm(centroid1, ord=1)
manNorm9 = np.linalg.norm(centroid9, ord=1)

euNorm0 = np.linalg.norm(centroid0, ord=2) #2-norms aka euclidean norm versions of each digit
euNorm1 = np.linalg.norm(centroid1, ord=2)
euNorm9 = np.linalg.norm(centroid9, ord=2)

cubeNorm0 = np.linalg.norm(centroid0, ord=3) #3-norm aka cubed norm versions of each digit
cubeNorm1 = np.linalg.norm(centroid1, ord=3)
cubeNorm9 = np.linalg.norm(centroid9, ord=3)


print("manhattan norms")
print("digit 0:", manNorm0)
print("digit 1:", manNorm1)
print("digit 9:", manNorm9)

print("euclidean norms")
print("digit 0:", euNorm0)
print("digit 1:", euNorm1)
print("digit 9:", euNorm9)

print("3-norms")
print("digit 0:", cubeNorm0)
print("digit 1:", cubeNorm1)
print("digit 9:", cubeNorm9)
