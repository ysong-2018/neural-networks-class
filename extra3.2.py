#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:10:03 2018

@author: yangsong
"""

import random
import numpy as np
from numpy  import array
#import matplotlib.pyplot
import pylab
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls

#I will just let the separating line be y=x

#  fraction of D where f and h disagree. n counts where they disagree, iterate through every one of them
def E_in(w,dataset):
	x_0 = 1
	n = 0
	for raw in dataset:
		y = raw[-1]
		if np.dot(w,np.array([x_0,raw[0],raw[1]])) <0:
			if y != -1:
				n = n+1
		else:
			if y!= 1:
				n = n+1
	return n/dataset.shape[0]


def misclassified(w,dataset):
	x_0 = 1
	for row in dataset:
		y = row[-1]
		x = np.array([x_0,row[0],row[1]])
		if (y*np.dot(w,x)) <= 0:
			break
	return x,y


# Input dataset should be length 3 and in form [[x1,x2,y],......].Has information for each point 
# weights is an initial weight
# T is the number of iterations
def pocket_new(testdataset,dataset,weights, T):
    # generate a new testing set
    #testdataset = generate_data()
    #print(dataset)
    #print(testdataset)
    
    w = np.array(weights,ndmin=1).T    # .T means transpose. this is used for later steps in np.dot
    best_w = w
    best_error = E_in(w, dataset)
    #best_error_test = E_in(w, testdataset)


    E_w = np.array([best_error])
    E_wop = np.array([best_error])
    
    #E_w_test = np.array([best_error_test])
    #E_wop_test =  np.array([best_error_test])
    
    
    
    

    for t in range(0,T-1):                  # for t=0,...,T-1 do
        x,y = misclassified(w,dataset)
        w = w + y*x                        # run PLA for on update to obtain w(t+1)
        error = E_in(w,dataset)
        #testerror = E_in(w,testdataset)
        if error < best_error:              # if w(t+1) is better than w, set w to w(t+1)
            best_w = w                     # best_w is the w hat returned. 
            best_error = error
        #if testerror < best_error_test:
        #    best_error_test = testerror
        E_w = np.append(E_w,np.array([error]),axis=0)
        E_wop = np.append(E_wop,np.array([best_error]),axis=0)
        
        #E_w_test = np.append(E_w_test,np.array([testerror]),axis=0)
        #E_wop_test = np.append(E_wop_test,np.array([best_error_test]),axis=0)
        
        
    return [best_w,E_w,E_wop]

def rand():
    while True:
        
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        return [x1,x2]

def rand_select():
    return random.randint(0,99)

def data_init():
    # I choose a random line in d=2. Let it be 0+x1-x2=0
    dataset = []
    x1list=[]
    x2list=[]
    # loop through N =100 and get initial dataset
    for i in range(100):
    #    x1 = random.randint(-100,101)
    #    x2 = random.randint(-100,101)
        [x1,x2] = rand()
        if (x1-x2>=0):
            y = +1
        else:
            y = -1
            
        #print(x1,x2,y)
        currentpoint = [x1,x2,y]
        x1list.append(x1)
        x2list.append(x2)
        dataset.append(currentpoint)
        
    #print(dataset)
    return dataset


def data_flip(dataset):
    # select N/10=10 points and flip the labels
    selectionindex=[]
    for i in range(10):
        select = rand_select()
        if not (select in selectionindex):
            selectionindex.append(select)
    #print("---------------------------------")   
    #print(selectionindex)
    
    
    for idx in selectionindex:
        dataset[idx][2] = - dataset[idx][2]
    #print(dataset)
    
    dataset = np.asarray(dataset)
    return dataset

def generate_data():
    dataset = data_init()
    
    dataset = data_flip(dataset)
    return dataset


if __name__ == "__main__":
    ## main function
    
    #dataset = generate_data()
    
   
    # let weights =[1,1,1]
    
    weights =[1,1,1]
    T = 1000
    testdataset = generate_data()
    total_best_w=[]
    total_E_w=[]
    total_E_wop=[]
    total_E_w_test=[]
    total_E_wop_test=[]
    for loop in range(20):
        dataset = generate_data()
        
        [best_w,E_w,E_wop] = pocket_new(testdataset,dataset,weights, T)
        total_best_w.append(best_w.tolist())
        total_E_w.append(E_w.tolist())
        total_E_wop.append(E_wop.tolist())
        #total_E_w_test.append(E_w_test.tolist())
        #total_E_wop_test.append(E_wop_test.tolist())
        
    avg_total_E_w=[]
    avg_total_E_wop=[]
    #avg_total_E_w_test=[]
    #avg_total_E_wop_test=[]
    
    for i in range(1000):
        avg_E_w=0
        for j1 in range(20):
            avg_E_w += total_E_w[j1][i]
        avg_E_w /= 20
        avg_total_E_w.append(avg_E_w)
        
        avg_E_wop=0
        for j1 in range(20):
            avg_E_wop += total_E_wop[j1][i]
        avg_E_wop /= 20
        avg_total_E_wop.append(avg_E_wop)
    
#        avg_E_w_test=0
#        for j1 in range(20):
#            avg_E_w_test += total_E_w_test[j1][i]
#        avg_E_w_test /= 20
#        avg_total_E_w_test.append(avg_E_w_test)
#    
#        avg_E_wop_test=0
#        for j1 in range(20):
#            avg_E_wop_test += total_E_wop_test[j1][i]
#        avg_E_wop /= 20
#        avg_total_E_wop_test.append(avg_E_wop)
    
    # put the xaxis
    xaxis = []
    for i in range(1000):
        xaxis.append(i+1)
        
    #sqrt((2/1000)*ln(1000)) = 0.11753
    avg_total_E_w_test = [(i+0.11753) for i in avg_total_E_w]
    avg_total_E_wop_test = [(i+0.11753) for i in avg_total_E_wop]
    
    
    
    colors = ['b', 'c', 'y', 'm']
    
    
    
    plt.scatter(xaxis,avg_total_E_w,marker = ".",color=colors[0], label="E_in")
    plt.scatter(xaxis,avg_total_E_wop,marker = ".",color=colors[1],label="E_in_best")
    plt.scatter(xaxis,avg_total_E_w_test,marker = ".",color=colors[2],label="E_out")
    plt.scatter(xaxis,avg_total_E_wop_test,marker = ".",color=colors[3],label="E_out_best")
    

    plt.legend()
    plt.show()
    
