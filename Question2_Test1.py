# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:51:18 2020

@author: vman9
"""


import numpy as np
import random


#Taking random values for first data set
a = np.random.uniform(-0.5,0.5,4)
b = np.random.uniform(-0.5,0.5,4)

print(a)
print(b)
##c = np.random.uniform(-3,-4,4)
##d = np.random.uniform(3,4,4)

##c = [3,3,-3,-3]
##d = [3,-3,3,-3]

c=[]
d=[]
count1=0

#Generating random values between the circles of radius 3 and 4
while(count1<4):
    l = random.uniform(-4.0,4)
    m = random.uniform(-4.0,4)
    n = np.square(l)+np.square(m)
    if(n > 9.0 and n < 16.0):
        count1 = count1 + 1
        c.append(l)
        d.append(m)

#Appending the values generated in the previous step to the original data set
x1 = np.hstack((a,c))
x2 = np.hstack((b,d))
x3 = [0,0,0,0,0,0,0,0]

#  input x3
for i in range(len(x1)):
    x3[i] = np.square(x1[i]) + np.square(x2[i])

print(x1)
print(x2)
print(x3)
    
# x = np.vstack((x1,x2,x3))
# # print(x)

# x = np.array([x1,x2,x3])
# print(x)

x = [[],[],[],[],[],[],[],[]]
for i in range(8):
    x[i].append(x1[i])
    x[i].append(x2[i])
    x[i].append(x3[i])
x = np.array(x)
print(x)

#  expected output
t =  [1,1,1,1,0,0,0,0]

w1 = np.array([[random.uniform(-2.0,2)for i in range(3)]for i in range(3)])
print(w1)

b1 = np.array([random.uniform(-2.0,2)for i in range(3)])
print(b1)

w2 = np.array([random.uniform(-2.0,2)for i in range(3)])
b2 = random.uniform(-2.0,2)
print(w2)
print(b2)

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of Sigmoid function 
def der_sigmoid(x):
    return x*(1-x)


lr = 0.1
count = 0

output=[]
while(count<10000):
    count=count+1
    print(count)
    for i in range(0,8):
        y = x[i].dot(w1.T)+b1
#         print(y)
        sigmoid_list = np.array([])
        for j in range(3):
            sigmoid_list = np.append(sigmoid_list,sigmoid(y[j]))
#         print(t[i])
        z = sigmoid_list.dot(w2) + b2
#         print("Z: ",z)
        z_sigmoid = sigmoid(z)
#         print("Sigmoid of Z: ",z_sigmoid)
#         print("-------------------------------------------------------------------------")
#         error = 0
        der_error = t[i]-z_sigmoid
        dw2 = np.multiply(lr*der_error*der_sigmoid(z_sigmoid),y)
        db2 = lr*der_error*der_sigmoid(z_sigmoid)
#         print("dw2: ",dw2)
#         print("db2: ",db2)
        
        test = np.multiply(der_error*der_sigmoid(z_sigmoid), w2)
#         print("Alpha: ",test)
        
        test_sum = np.sum(test)
#         print("Alpha_Sum: ",test_sum)
        
        dw1 = np.array([[random.uniform(0,0)for i in range(3)]for j in range(3)])
#         print(dw1)
        
        for m in range(3):
            for n in range(3):
                dw1[m][n] = lr * x[i][n]*der_sigmoid(sigmoid_list[m])*test_sum
        
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
    
    
    # put error in between 0.7 and 0.9 and try the program
    if(error > 0.1):
        output = []
    else:
#         print("aaaaaaaaaa")
        print("Input: ",x)
        print("Weights1: {} \n Weights2: {} \n Bias1: {} \n Bias2: {}".format(w1,w2,b1,b2))
        print("Output: ",output)
        break;
           
    print("\n\n")
#     print(sigmoid_list)
# print(output)