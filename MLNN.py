import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import sklearn
from sklearn.model_selection import train_test_split
from itertools import permutations 
from scipy import stats
import random



def logistic(beta, x):
    return 1/(1 + np.exp(-beta*x))
    
def tanh(beta, x):
    return (np.exp(beta*x) - np.exp(beta*x))/(np.exp(beta*x) + np.exp(-beta*x))

def linear(beta, x):
    return x

def softmax(beta, x):
    return (np.exp(x.T)/sum(np.exp(x.T))).T


def act(actFun, vec, β):
    if(actFun == 'logistic'):
        return logistic(β, vec)
    elif(actFun == 'tanh'):
        return tanh(β, vec)
    elif(actFun == 'linear'):
        return linear(β, vec)
    elif(actFun == 'softmax'):
        return softmax(β, vec)
    else:
        print('No such activation function exists')
        return
    
    
def dlogistic(beta,x):
    return beta*logistic(beta,x)*(1-logistic(beta,x))

def dlinear(beta, x):
    return 1

def dtanh(beta, x):
    return beta*(1 - tanh(beta, x)**2)

def dact(actFun, vec, β):
    if(actFun == 'logistic'):
        return dlogistic(β, vec)
    elif(actFun == 'linear'):
        return dlinear(β, vec)
    elif(actFun == 'tanh'):
        return dtanh(β, vec)
    else:
        print('No such activation function exists')
        return
    


class MLNN(object):
    
    def __init__(self, nodes, actFuns, low, high):
        
        self.nodes = nodes
        self.actFuns = actFuns
        weights = []
        
        for i in range(1,len(nodes)):
            
            A = np.random.uniform(low*(i+1),high*(i+1),(nodes[i], nodes[i-1]+1))
            A[:,:1] = 0
            weights.append(A)
        
        self.weights = weights
        self.layers = len(nodes)
        
    def fit(self, Xdata, Ydata, η, α, β, epochs):
        
        Nx, d = Xdata.shape
        Ny, k = Ydata.shape
        
        assert(d == self.nodes[0])
        assert(k == self.nodes[-1])
        assert(Nx == Ny)
        
        N = Nx
        
        o = np.ones(N)
        o = np.reshape(o, (N, 1))
        
        S = [0]*(self.layers)
        S[0] = Xdata
        A = [0]*(self.layers-1)
        δweights = [0]*(self.layers-1)
        
        avgErrors = []
        
        for i in range(epochs):
            
            err = 0
            
            for h in range(self.layers-1):
        
                dth = np.hstack((o, S[h]))
                
                Ah = np.dot(dth, self.weights[h].T)
                Sh = act(self.actFuns[h], Ah, β)
                
                A[h] = Ah
                S[h+1] = Sh
                
            if(self.actFuns[-1] == 'softmax'):
                err = (-np.log(np.sum(np.multiply(Ydata, S[-1]), axis = 1)))/(2*N)
            else:
                err = ((Ydata-S[-1])**2)/(2*N)
            avgErrors.append(np.sum(err))
            
            δ = [0]*(self.layers-1)
            
            if(self.actFuns[-1] == 'softmax'):
                δ[-1] = np.multiply((1 - S[-1]), Ydata)
            else:
                δ[-1] = np.multiply((Ydata-S[-1]), dact(self.actFuns[-1], A[-1], β))
            
            
            for h in range(self.layers-3, -1, -1):
                
                δ[h] = np.multiply(np.dot(δ[h+1],self.weights[h+1][:,1:]), dact(self.actFuns[h], A[h], β))
                
                
            for h in range(self.layers-1):
                
                δw =  (η/N)*np.dot(δ[h].T, np.hstack((o, S[h])))
                self.weights[h] += (δw + α*δweights[h])
                δweights[h] = δw
    
            print('echo %d, error = %.2f'%(i,avgErrors[-1])) 
                
        plt.plot(list(range(1, epochs + 1, 1)), avgErrors)
        plt.title('Epochs vs Avg Error')
        plt.show()
             
        return self.weights, avgErrors
    
    
    def predict(self, data, β):
        
        N, d = data.shape
        
        o = np.ones(N)
        o = np.reshape(o, (N, 1))
        
        S = []
        S.append(data)
        
        for h in range(self.layers-1):
            
            dth = np.hstack((o, S[h]))
                
            Ah = np.dot(dth, self.weights[h].T)
            Sh = act(self.actFuns[h], Ah, β)
        
            S.append(Sh)
        
        return S[-1]