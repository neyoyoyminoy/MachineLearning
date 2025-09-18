'''
Brendan Nellis     ujf306
Intro to Machine Learning      Fall 2025
Homework 1
Create a noisy version Xn by corrupting each pixel independently with probability 5%, replacing it with 0 or 255.
Perform SVD on Xn and construct low-rank approximations:
Yk = QkQk'Xn
kset := {1, 51, 101, . . . , 784}

Compute: Compute errors e(X, YK) and e(Xn, YK).
Plot: Plot e(Xn, YK) and e(X, YK) versus K, including a benchmark e(X, Xn).
'''


import numpy as np

'''the next six lines are from problem 0'''
from tensorflow.keras.datasets import mnist #given from Dr. Markopoulos

(trainX, trainY), _ = mnist.load_data() #this will load MNIST data into the variables, leaving the 'test' set blank for simplicity
mask = np.isin(trainY, [0,1,9]) #extracts only 0, 1, and 9 for analysis
trainingImages = trainX[mask] #grabbing the masked images for vectorization
vTrainingImages = trainingImages.reshape(trainingImages.shape[0], -1) #flattens/reshapes (matrix vectorization) images of 0, 1, and 9; the '-1'
X = vTrainingImages.T.astype('float32') #fixed it to columns with transpose 'T', sets matrix X to float32

noisyMask = np.random.binomial(n = 1, p = 0.05, size = X.shape).astype(bool)  #starting at 1, probability of 5%, and the max vector size of the MNIST is 748 (28*28); would not work without 'bool' type

noisyValues = np.random.choice([0, 255], size = X.shape) #dependent on noisyMask; change the corrupted pixels to a '0' or '255'

Xn = np.where(noisyMask, noisyValues, X) #replace corrupted pixels with their noisyValues

'''
print("X: ", Xn)
print("Xn: ", noisyXn)
'''

U, S, Vh = np.linalg.svd(Xn, full_matrices = False) #performing SVD on Xn

'''
print("SVD")
print("U:", U.shape) #left
print("S:", S.shape) #singular
print("Vh:", Vh.shape) #right
'''

kSet = [1, 51, 101, 784] #from problem 3

for k in kSet:
  Qk = U[:, :k] #using first 'k' columns of 'U'
  projection = np.dot(Qk.T, Xn) #this was hard but used the dot product and transpose to get the projection
  Yk = np.dot(Qk, projection)

  '''
  print("K: ", K)
  print("Yk: ", Yk)
  '''

error1 = np.linalg.norm(X - Yk, 'fro') #this is basically the clean error; Frobenius norm for both errors aka 'fro'
error2 = np.linalg.norm(Xn - Yk, 'fro') #so this is the noisy error

'''
print("error1: ", error1)
print("error2: ", error2)
'''

import matplotlib.pyplot as plt

benchmarkError = np.linalg.norm(X - Xn, 'fro') #the defined benchmark from Dr. Markopoulos

plt.plot(kSet, error1, label = "e(X, Yk)") #clean plot
plt.plot(kSet, error2, label = "e(Xn, Yk)") # noisy plot
plt.axhline(y = benchmarkError, linewidth = 4, color = "b", label = "e(X, Xn)") #should make benchmark error line thick and blue

plt.xlabel("k") #labels
plt.ylabel("error")
plt.title("errors versus k")
plt.legend()
plt.show()
