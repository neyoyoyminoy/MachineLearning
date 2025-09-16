#Brendan Nellis   ujf306
#Intro to Machine Learning  Fall 2025
#Homework 1
#Create a noisy version Xn by corrupting each pixel independently with probability 5%, replacing it with 0 or 255. Perform SVD on Xn and construct low-rank approximations
#Yk = QkQk'Xn
#kset := {1, 51, 101, . . . , 784}

import numpy as np

noisyXn = np.random.binomial(n = 1, p = 0.05, size = 748) #starting at 1, probability of 5%, and the max vector size of the MNIST is 748 (28*28)

#x[x == 1] = 250

print(noisyXn)

'''n = 0

while n <= 784:
  n += 1
  print(n)
else:
  print('loop stopped')
 '''
