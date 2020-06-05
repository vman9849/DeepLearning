# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:15:35 2020

@author: vman9
"""


import numpy as np
import random

x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
t = np.array([0,0,0,1])

# x = np.array([
#  [0, 0, 0],
#  [0, 1, 1],
#  [1, 0, 1],
#  [1, 1, 1],
#  [1, 1, 0],
#  [4, 0, 4],
#  [0, 4, 4],
#  [4, 4, 4]])


# t= np.array([1,1,1,1,1,0,0,0])

w1 = np.array([[random.uniform(-2.0,2)for i in range(3)]for j in range(3)])
# print(w1)

b1 = np.array([random.uniform(-2.0,2)for i in range(3)])
# print(b1)

w2 = np.array([random.uniform(-2.0,2)for i in range(3)])
b2 = random.uniform(-2.0,2)
# print(w2)
# print(b2)

lr = 0.1
count = 0
output=[]

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of Sigmoid function 
def der_sigmoid(x):
    return x*(1-x)


while(count<10000):
    count=count+1
    print("Iteration: ",count)
    for i in range(0,4):
        y = x[i].dot(w1.T)+b1
        # print(y)
        sigmoid_list = np.array([])
        for j in range(3):
            sigmoid_list = np.append(sigmoid_list,sigmoid(y[j]))
        # print("Sigmoid List: ",sigmoid_list)
        # print(t[i])
        z = sigmoid_list.dot(w2) + b2
        # print("Z: ",z)
        z_sigmoid = sigmoid(z)
        # print("Sigmoid of Z: ",z_sigmoid)
#         print("-------------------------------------------------------------------------")
#         error = 0
        der_error = t[i]-z_sigmoid
        dw2 = np.multiply(lr*der_error*der_sigmoid(z_sigmoid),sigmoid_list)
        db2 = lr*der_error*der_sigmoid(z_sigmoid)
#         print("dw2: ",dw2)
#         print("db2: ",db2)
        
        test = np.multiply(der_error*der_sigmoid(z_sigmoid), w2)
#         print("Sum: ",test)
        
        test_sum = np.sum(test)
#         print("Test_sum: ",test_sum)
        
        dw1 = np.array([[random.uniform(0,0)for i in range(3)]for j in range(3)])
#         print(dw1)
        
        for m in range(3):
            for n in range(3):
                dw1[m][n] = lr * x[i][n]*der_sigmoid(sigmoid_list[m])*test_sum
                # print("Differentiation in weights: ",dw1[m][n])
        
        sigmoid_der_list = np.array([])
        for i in range(3):
            sigmoid_der_list = np.append(sigmoid_der_list, der_sigmoid(sigmoid_list[i]))
#         print("Sigmoid_der list: ",sigmoid_der_list)
        
        db1 = lr*sigmoid_der_list*test_sum
        
        w1 = np.add(w1,dw1)
        b1 =np.add(b1, db1)
        
        w2 = np.add(w2,dw2)
        b2 = np.add(b2,db2)
        
        output.insert(i,z_sigmoid)
        
    error = 0
    for s in range(len(output)):
        error = error + np.square(output[s] - t[s])
    print("Error: ",error)       
    
        
    if(error > 0.7):
        output = []
    else:
#         print("ccccccccccc")    
        print("Input: ",x)
        print("Weights1: {} \n Weights2: {} \n Bias1: {} \n Bias2: {}".format(w1,w2,b1,b2))
        print("Output: ",output)
        break;

        
        
    print("\n\n")
#     print(sigmoid_list)
# print(output)