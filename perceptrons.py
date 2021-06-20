import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from itertools import permutations 
from scipy import stats
import random

class Perceptron(object):
    
    def __init__(self, weights, dim, actFun):
        
        self.dim = dim
        init_weights = np.random.normal(0, 1, dim+1)
        self.weights = init_weights
        self.actFun = actFun
        
        
    def fitTwo(self, trainData, lrRate, β, epochs):
        
        if(self.actFun == 'logistic' or self.actFun == 'tanh'):
            avgErrors = []
        else:
            print('Error\nActivation Function Not Available\nUse either logistic or tanh')
            return 
    
        
        for i in range(epochs):
            
            avgError = 0
            
            for k in range(2):
                
                l = len(trainData[k])
                o = np.ones(l)
                o = np.reshape(o, (l, 1))
                dt = np.hstack((o, trainData[k]))

                r = list(range(l))
                random.shuffle(r)
                
                
                for j in r:
                    
                    a = np.sum(self.weights*dt[j])
                    
                    if(self.actFun == 'logistic'):
                        
                        s = 1/(1 + np.exp(-β*a))
                        e = np.sum(0.5*((k-s)**2))
                        
                        δw = lrRate*(k-s)*β*s*(1-s)*dt[j]
                        
                    elif(self.actFun == 'tanh'):
                        
                        x = np.exp(β*a)
                        y = np.exp(-β*a)
                        s = (x-y)/(x+y)
                        
                        e = np.sum(0.5*((k-s)**2))
                        
                        δw = lrRate*(k-s)*β*(1 - s**2)*dt[j]
                    
                        
                    self.weights += δw
                    avgError += e
                
            avgErrors.append(avgError)
                    
        plt.plot(list(range(1, epochs + 1, 1)), avgErrors)
        plt.title('Epochs vs Avg Error')
        plt.show()
        
        return self.weights
    

        
class MultiPerceptrons(object):
    
    def __init__(self, weights, dim, actFun):
        
        self.dim = dim
        init_weights = np.random.normal(0, 1, dim+1)
        self.weights = init_weights
        self.actFun = actFun
        
        
    def lexicographical_permutation(self, k): 
    
        string = ''
    
        for i in range(k):
            string = string + str(i)
        
        perm = sorted(''.join(chars) for chars in permutations(string)) 
        ans = []
        for x in perm: 
            a = (int(x[0]), int(x[1]))
            ans.append(tuple(sorted(a)))
        
        return set(ans)
        
        
    def fit(self, trainData, η, β, epochs, N):

        self.beta = β
        
        K = int(N*(N-1)/2)
        W = [[0 for x in range(N)] for x in range(N)] 
        
        ans = self.lexicographical_permutation(K)
        
        for i in ans:
            
            print('For Classes', i[0], i[1])
            
            perceptron = Perceptron(self.weights, self.dim, self.actFun)
            w = perceptron.fitTwo([trainData[i[0]], trainData[i[1]]], η, β, epochs)
            
            W[i[0]][i[1]] = w
            W[i[1]][i[0]] = w
            
            
        return W


    def logistic(self, x):
        return 1/(1 + np.exp(-self.beta*x))
    
    def tanh(self, x):
        return (np.exp(self.beta*x) - np.exp(-self.beta*x))/(np.exp(self.beta*x) + np.exp(-self.beta*x))
    
    
    def accuracy(self, finalWeights, data, N):
        
        if(self.actFun == 'logistic'):
            
            truePredCount = 0
            total_count = 0
            
            for k in range(N):
    
                labels = []
    
                l = len(data[k])
                o = np.ones(l)
                o = np.reshape(o, (l, 1))
                dt = np.hstack((o, data[k]))
                total_count += l
    
                for i in range(N):
                    for j in range(N):
                        if(i < j):
                
                            A = np.multiply(self.logistic(np.sum(finalWeights[i][j]*dt, axis = 1)) > 0.5, j+1)
                            A += np.multiply(self.logistic(np.sum(finalWeights[i][j]*dt, axis = 1)) <= 0.5, i+1)
                            A = np.reshape(A, (l, 1))
                
                            if(labels == []):
                                labels = A
                    
                            else:
                                labels = np.hstack((labels, A))   
                    
                preds = stats.mode(labels, axis = 1)[0]          
                truePredCount += np.sum(np.multiply(preds == k+1, 1))
    
            return (truePredCount/total_count)*100
        
        elif(self.actFun == 'tanh'):
            
            truePredCount = 0
            total_count = 0
            
            for k in range(N):
    
                labels = []
    
                l = len(data[k])
                o = np.ones(l)
                o = np.reshape(o, (l, 1))
                dt = np.hstack((o, data[k]))
                total_count += l
    
                for i in range(N):
                    for j in range(N):
                        if(i < j):
                
                            A = np.multiply(self.tanh(np.sum(finalWeights[i][j]*dt, axis = 1)) > 0, j+1)
                            A += np.multiply(self.tanh(np.sum(finalWeights[i][j]*dt, axis = 1)) <= 0, i+1)
                            A = np.reshape(A, (l, 1))
                
                            if(labels == []):
                                labels = A
                    
                            else:
                                labels = np.hstack((labels, A))   
                    
                preds = stats.mode(labels, axis = 1)[0]          
                truePredCount += np.sum(np.multiply(preds == k+1, 1))
    
            return (truePredCount/total_count)*100

    def decisionPlot(self, finalWeights, min, max, N, trainData):
        
        x = np.arange(min, max, 0.1)
        y = np.arange(min, max, 0.1)
        xx, yy = np.meshgrid(x, y)
        l = len(x)*len(x)
        o = np.ones(l)
        o = np.reshape(o, (l, 1))
        dt = np.hstack((np.reshape(xx, (l, 1)), np.reshape(yy, (l, 1))))
        dt = np.hstack((o, dt))

        labels = []

        for i in range(N):
            for j in range(N):
                if(i < j):

                    if(self.actFun == 'logistic'):
                        A = np.multiply(self.logistic(np.sum(finalWeights[i][j]*dt, axis = 1)) > 0.5, j+1)
                        A += np.multiply(self.logistic(np.sum(finalWeights[i][j]*dt, axis = 1)) <= 0.5, i+1)

                    elif(self.actFun == 'tanh'):
                        A = np.multiply(self.tanh(np.sum(finalWeights[i][j]*dt, axis = 1)) > 0, j+1)
                        A += np.multiply(self.tanh(np.sum(finalWeights[i][j]*dt, axis = 1)) <= 0, i+1)

                    A = np.reshape(A, (l, 1))
                
                    if(labels == []):
                        labels = A
                    
                    else:
                        labels = np.hstack((labels, A))   
                    
        preds = stats.mode(labels, axis = 1)[0]
        plt.scatter(dt[:,1].reshape(-1,1),dt[:,2].reshape(-1,1), c = preds, alpha = 0.1)

        for i in range(N):
            plt.scatter(trainData[i][:,0], trainData[i][:,1], s = 1)
            
        plt.show()