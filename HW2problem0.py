'''
Problem 0 (0 points)
Load arrays A, y, and  𝑥𝑖𝑛𝑖𝑡  from the csv files provided in this assignment.
Denote by N and K the rows and columns of A, respectively.
Consider the least squares (LS) problem  𝑥𝑜𝑝𝑡  = arg  𝑚𝑖𝑛𝑥∈𝑅𝐾  L(x), where L(x) = ∥Ax − y∥ 2/2.
'''

import numpy as np
import os
import pandas as pd

#mounting drive
from google.colab import drive  #this just has to be given permission once
drive.mount('/content/drive')

folderPath = '/content/drive/MyDrive/MachineLearning/HW2_CSV_DATA' #this is the folder path with google drive

#loading array A from included .csv file
aPath = os.path.join(folderPath, 'HW2_A.csv')
A = pd.read_csv(aPath, header=None, delimiter=' ').values.astype(float) #making sure all values are numeric for later on


#loading array y
yPath = os.path.join(folderPath, 'HW2_y.csv')
y = pd.read_csv(yPath, header=None, delimiter=' ').values.flatten()

#load array xinit
xInitPath = os.path.join(folderPath, 'HW2_xinit.csv')
xInit = pd.read_csv(xInitPath, header=None, delimiter=' ').values.flatten()

N, K = A.shape #N rows and K columns of A

print(f"array A: {A.shape}")
print(f"array y: {y.shape}")
print(f"array xinit: {xInit.shape}")
print(f"{N} rows and {K} columns of A")
