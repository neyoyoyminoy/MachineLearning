'''
Brendan Nellis     ujf306
Intro to Machine Learning      Fall 2025
Homework 1
Case 1: G = A, f = yA
Case 2: G = B, f = yB
Case 3: G = B, f = yB2
Case 4: G = C, f = yC
Case 5: G = C, f = yC2
Case 6: G = D, f = yD
Case 7: G = D, f = yD2

Compute: Compute the size and rank of matrix G.
Compute: Compute the projection matrix for the span of G and determine if f is in the span of G.
Discuss: Is the system f = Gw overdetermined or underdetermined?
Discuss: Does f = Gw have none, one, or infinitely many solutions?
Compute: Compute a solution if one exists.
'''


import numpy as np

A = np.array([[-2.74125009, 2.24215689, -0.60553211, -0.16755625],
              [-0.34868395, 0.29538923, -0.45259498, 0.50015934],
              [2.49664208, 0.27798324, 2.00739274, 0.2197803]])
B = np.array([[-2.74125009 , -0.34868395, 2.49664208],
              [2.24215689, 0.29538923, 0.27798324],
              [-0.60553211, -0.45259498, 2.00739274],
              [-0.16755625, 0.50015934, 0.2197803]])
C = np.array([[0.31997336, 0.43316234, -0.33457014, -0.34017903],
              [1.12969075, 1.52931319, -1.18122581, -1.20102843],
              [0.2008776, 0.27193705, -0.21004138, -0.21356262]])
D = np.array([[0.07999334 , 0.28242269, 0.0502194],
              [0.10829058, 0.3823283, 0.06798426],
              [-0.08364254, -0.29530645, -0.05251035],
              [-0.08504476, -0.30025711, -0.05339065]])

yA = np.array([[0.61339829],
               [0.11012282],
               [-0.06426754]])
yB = np.array([[0.66761214],
               [0.35931116],
               [0.74289966],
               [0.02979187]])
yB2 = np.array([[0.24982762],
               [-0.45768269],
               [0.22778277],
               [0.6341392]])
yC = np.array([[0.1421664],
               [0.50192948],
               [0.08925132]])
yC2 = np.array([[-1.01480112],
               [0.4115211],
               [-0.45229071]])
yD = np.array([[0.41615372],
               [0.56336601],
               [-0.43513813],
               [-0.44243299]])
yD2 = np.array([[0.47277025],
               [-0.64357627],
               [1.30059591],
               [1.426948]])

case = [("case 1..", A, yA),("case 2..", B, yB),("case 3..", B, yB2),("case 4..", C, yC),("case 5..", C, yC2),("case 6..", D, yD),("case 7..", D, yD2)]

for caseName, G, f in case:
  print(caseName)
  print("size of G: ", G.size)
  print("shape of G: ", G.shape)

  rankG = np.linalg.matrix_rank(G)
  print("rank of G: ", rankG)

  projectionG = G @ np.linalg.pinv(G) #had to use psudoinverse since the first couple of tests would have an error on case 4
  projectionGf = np.dot(projectionG, f) #projection of vector f onto matrix G
  print(projectionGf)

  if np.allclose(projectionGf, f): #allclose is a comparison tool with a tolarance aka the absolute difference
    print("f is in the span of G")
  else:
    print("f is not in the span of G")
  
  w = np.linalg.lstsq(G, f, rcond=None)[0] #least squares used to find the solution if one exists
  print("the solution to f = Gw is... ", w)

  print("\n")
