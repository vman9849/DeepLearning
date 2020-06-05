# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:57:06 2020

@author: vman9
"""

import numpy as np
import random

x = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
t = [0,0,0,1]

w1 = [[random.uniform(-2.0,2)for i in range(3)]for j in range(3)]
# print(w1)
b1 = np.array([random.uniform(-2.0,2)for i in range(3)])
# print(b1)

# weights2 generated
w2 = [[random.uniform(-2.0,2)for i in range(3)]for j in range(3)]
# w2 = [random.uniform(-2.0,2)for i in range(3)]
# print(w2)

#bias 2 generated
b2 = random.uniform(-2.0,2)
# print(b2)
    
#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of Sigmoid function 
def der_sigmoid(x):
    return x*(1-x)

lr = 0.1
count = 0
result = []

dw1 = [[0 for i in range(3)]for j in range(3)]
dw2 = [[0 for i in range(3)]for j in range(3)]

while(count<1000000):
    count = count+1
    print("Count: ",count)
    for i in range(0,3):
        for n in range(0,3):
            v1 = x[0][n] * w1[0][0] + x[1][n] * w1[1][0] + x[2][n] * w1[2][0] + b1[0]
            v2 = x[0][n] * w1[0][1] + x[1][n] * w1[1][1] + x[2][n] * w1[2][1] + b1[1]
            v3 = x[0][n] * w1[0][2] + x[1][n] * w1[1][2] + x[2][n] * w1[2][2] + b1[2]
        
        u = []
        u.append(sigmoid(v1))
        u.append(sigmoid(v2))
        u.append(sigmoid(v3))
        # print(u)

        z = []
        # print(z)
        z1 = u[0]*w2[0][0] + u[1]*w2[1][0] + u[2]*w2[2][0] + b1[0]
        z2 = u[0]*w2[0][1] + u[1]*w2[1][1] + u[2]*w2[2][1] + b1[1]
        z3 = u[0]*w2[0][2] + u[1]*w2[1][2] + u[2]*w2[2][2] + b1[2]
        
        p = []
        p.append(sigmoid(z1))
        p.append(sigmoid(z2))
        p.append(sigmoid(z3))
        # print(p)
        
        y = p[0]*w2[0][0] + p[1]*w2[0][1] + p[2]*w2[0][2] + b2
        # y = u[0]*w2[0][0] + u[1]*w2[0][1] + u[2]*w2[0][2] + b2
        # print(y)
        
        s = sigmoid(y)
        
        result.insert(i,s)
        
        summation = 0
        # summation = w2[0]*s*(1-s)*(t[i]-s) + w2[1]*s*(1-s)*(t[i]-s) + w2[2]*s*(1-s)*(t[i]-s)
        summation = w2[0][0]*s*(1-s)*(t[i]-s) + w2[0][1]*s*(1-s)*(t[i]-s) + w2[0][2]*s*(1-s)*(t[i]-s)
        # print(summation)
        
        
        for k in range(3):
            for j in range(3):
                dw2[k][j] = lr*x[i][j]*u[k]*(1-u[k])*summation
        
        var = 0 
        for k in range (3):
            for j in range(3):
                # print(k)
                # print(j)
                # print(w2[k][j])
                # print(p[j])
                # print(1 - p[j])
                var= var + w2[k][j]*p[j]*(1-p[j])*summation
                
        for m in range(3):
            for n in range(3):
                dw1[m][n]= lr*x[m][i]*u[n]*(1-u[n])*var        
        
        db = [] 
        for k in range(3):
            db.insert(k,lr*u[i]*(1-u[i])*var)
        # dB1 = lr*u[0]*(1-u[0])*var
        # dB2 = lr*u[1]*(1-u[1])*var
        # dB3 = lr*u[2]*(1-u[2])*var
        
        for m in range(3):
          for n in range(3):
              w1[m][n] = w1[m][n] + dw1[m][n] 

        for k in range(3):
            b1[k] = b1[k] + db[k]
        
    error = 0
    for k in range(len(result)):
        error = error + (t[k] - result[k])**2
    print("Error: ",error,"\n")
    if(error < 0.1):
        print("Input: ",x)
        print("\n Weights1: ",w1)
        print("\n Weights2: ",w2)
        print("\Result: ",result)
        break
    else:
        result = []
    