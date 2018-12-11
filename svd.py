from numpy.linalg import svd
import numpy as np

movieRatings = [
    [2, 5, 3],
    [1, 2, 1],
    [4, 1, 1],
    [3, 5, 2],
    [5, 3, 1],
    [4, 5, 5],
    [2, 4, 2],
    [2, 2, 5],
]
"""U : Users, V : Values (Movie ratings per user, Each user gives 3 ratings), DF is just svd of the UxV matrix. """
U, singularValues, V = svd(movieRatings)
"""print "Original U, Singular Value of D_Frame, V:\n"
print "U : \n",U
print"Singular DF : \n ",singularValues
print "V : \n",V"""
print("Original User input array : ")
print (movieRatings)
Sigma = np.vstack([
    np.diag(singularValues),
    np.zeros((5, 3)),
])
print("\nSVD Implementation : ")
print(np.round(movieRatings - np.dot(U, np.dot(Sigma, V)), decimals=10))


U, singularValues, V = svd(movieRatings, full_matrices=False)
print("\nU :\n")
i=0;
for item in U:
    for value in item :
        if i>3:
            print(" ")
            i=0
        else:
            i+=1
            print(value,end=" ")

print ("\n\nD_Frame : \n"+str(singularValues))
print ("\nV : \n")
i=0
for item in V:
    for value in item:
        if i>3:
            print (" ")
            i=0
        else:
            i+=1
            print (value,end=" ")

Sigma = np.diag(singularValues)
print ("\n\nFinal Answer Matrix : \n")
print (np.round(movieRatings - np.dot(U, np.dot(Sigma, V.T)), decimals=10))