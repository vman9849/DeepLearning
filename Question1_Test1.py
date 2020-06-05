# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:15:35 2020

@author: vman9
"""


import numpy as np
import random

# Input values
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

# target values
t = np.array([0,0,0,1])

#weights generated
w1 = np.array([[random.uniform(-2.0,2)for i in range(3)]for j in range(3)])
# print(w1)

#bias 1 generated
b1 = np.array([random.uniform(-2.0,2)for i in range(3)])
# print(b1)

# weights2 generated
w2 = np.array([random.uniform(-2.0,2)for i in range(3)])

#bias 2 generated
b2 = random.uniform(-2.0,2)
# print(w2)
# print(b2)

learning_rate = 0.08
count = 0
output=[]

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of Sigmoid function 
def der_sigmoid(x):
    return x*(1-x)

# running the loop untill the  count becomes 10000 or the error is less than 0.07
while(count<100000):
    count=count+1
    print("Iteration: ",count)
    for i in range(0,4):
        y = x[i].dot(w1)+b1
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
        
        
        test = np.multiply(der_error*der_sigmoid(z_sigmoid), w2)
#         print("Sum: ",test)
        
        test_sum = np.sum(test)
#         print("Test_sum: ",test_sum)
        
        dw1 = np.array([[random.uniform(0,0)for i in range(3)]for j in range(3)])
#         print(dw1)

        for m in range(3):
            for n in range(3):
                dw1[m][n] = learning_rate * x[i][n]*der_sigmoid(sigmoid_list[m])*test_sum
                # print("Differentiation in weights: ",dw1[m][n])

        # calculating change in weights and calculating their equations 
        dw2 = np.multiply(sigmoid_list, learning_rate*der_error*der_sigmoid(z_sigmoid))
        db2 = learning_rate*der_error*der_sigmoid(z_sigmoid)
#         print("dw2: ",dw2)
#         print("db2: ",db2)
        

        sigmoid_der_list = np.array([])
        for i in range(3):
            sigmoid_der_list = np.append(sigmoid_der_list, der_sigmoid(sigmoid_list[i]))
#         print("Sigmoid_der list: ",sigmoid_der_list)
        
        db1 = learning_rate*sigmoid_der_list*test_sum
        
        w1 = np.add(w1,dw1)
        b1 = np.add(b1, db1)
        
        w2 = np.add(w2,dw2)
        b2 = np.add(b2,db2)
        
        output.insert(i,z_sigmoid)
        
    error = 0
    for s in range(len(output)):
        error = error + np.square(output[s] - t[s])
    print("Error: ",error)       
    
    #Test the error values between 0.7 and 0.9 the final values will be appearing     
    if(error > 0.5):
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