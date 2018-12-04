#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:59:42 2018

@author: Zi Xian Leong
"""
#################
# Simple Kriging
#################
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import *
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

rawdat = pd.read_csv('wmarker3_dat.csv', header=None, names=['X','Y','Z'] )
X = rawdat.iloc[:,0] # X Coordinates
X = np.array(X.values,dtype=np.float64) 
Y = rawdat.iloc[:,1] #Y Coordinates
Y = np.array(Y.values,dtype=np.float64)
Z = rawdat.iloc[:,2] # True Data Point
Z = np.array(Z.values,dtype=np.float64)

#Convariance Function
def Cov(x):
    a = 10
    return 1 - (abs(x) / a)

#Distances from one another
def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

dist_table = np.tile(None,(len(X),len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
       dist_table[i,j] = dist(X[i],Y[i],X[j],Y[j])
       
#Covariance of given data
cov_table = np.tile(None,(len(X),len(Y)))
for row in range(len(dist_table[0])):
    for col in range(len(dist_table[1])):
        cov_table[row,col] = Cov(dist_table[row,col])    

#Inverse Matrix
c_inv = inv(np.matrix(cov_table, dtype='float'))

#coordinates of target points
stride = 10 #take a target estimation point at every 10th pixel
x_mesh = []
for i in range(int(round(min(X))-stride), int(round(max(X)))+stride,stride):
    x_mesh.append(i)
    
y_mesh = []
for i in range(int(round(min(Y))-stride), int(round(max(Y)))+stride,stride):
    y_mesh.append(i)
    
#distance to target
canvas_dist = np.tile(None,(len(x_mesh),len(y_mesh)))

for itex in range(len(x_mesh)):
    for itey in range(len(y_mesh)):
        h_each=[]
        for i in range(len(X)):
            h_each.append(dist(X[i],Y[i],x_mesh[itex],y_mesh[itey])) #each target point's distance to given data, have 18 data points
        canvas_dist[itex][itey] = h_each
        
#Convariance of Target
cov_target_table = np.tile(None,(len(x_mesh),len(y_mesh)))
for i in range(len(x_mesh)):
    for j in range(len(y_mesh)):
        c_each =[]
        for each_h in range(len(canvas_dist[i][j])):
            c_each.append(Cov(canvas_dist[i][j][each_h]))
        cov_target_table[i][j]=c_each

#Weights
c_inv_array = np.asarray(c_inv)
weights = np.tile(None,(len(x_mesh),len(y_mesh)))
for i in range(len(x_mesh)):
    for j in range(len(y_mesh)):
        weights[i,j] = np.dot(c_inv_array, cov_target_table[i][j] )
#        weights[i,j] = np.dot(c_inv, cov_target_table[i][j] )
#mean of data
avg = sum(Z)/len(Z)

#Weight at zero (lambda0)
weight_sum = np.tile(None,(len(x_mesh),len(y_mesh)))
for i in range(len(x_mesh)):
    for j in range(len(y_mesh)):
        weight_sum[i,j] = sum(weights[i,j])
        
lambda0 = np.tile(None,(len(x_mesh),len(y_mesh)))
for i in range(len(x_mesh)):
    for j in range(len(y_mesh)):
        lambda0[i,j] = avg *(1 - weight_sum[i,j])
        
#Z prediction
        
weighted_sum = np.tile(None,(len(x_mesh),len(y_mesh)))
        
for i in range(len(x_mesh)):
    for j in range(len(y_mesh)):
        inside = []
        for k in range(len(X)):
            inside.append(weights[i][j][k]*Z[k])
        weighted_sum[i,j]=sum(inside)
                      
z_pred = np.tile(None,(len(x_mesh),len(y_mesh)))
for i in range(len(x_mesh)):
    for j in range(len(y_mesh)):
        z_pred[i,j]=lambda0[i,j]+weighted_sum[i,j]

z_pred = z_pred.astype(float)

##################################################
#2D Plot

plt.contourf(x_mesh,y_mesh,np.transpose(z_pred),30)
plt.colorbar()

for i in range(len(Z)):
    x = X[i]
    y = Y[i]
    plt.plot(x, y, 'ro')
    plt.text(x * (1 + 0.01), y * (1 + 0.01) , round(Z[i]), fontsize=8)

plt.xlim(min(x_mesh),max(x_mesh))
plt.ylim(min(y_mesh),max(y_mesh))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Kriging')
#plt.savefig('Fig/SK_2D.png',dpi=600)

##########################################################################
#3D Plot

#%matplotlib qt
#%matplotlib inline

#fig = plt.figure()
#ax = Axes3D(fig)
#x_mesh_g, y_mesh_g = np.meshgrid(x_mesh,y_mesh)
#surf = ax.plot_surface(x_mesh_g, y_mesh_g, np.transpose(z_pred))
#ax.view_init(10, -265)
#ax.scatter(X,Y,Z, color='red', s=25)
#ax.set_xlabel('X', fontsize=12)
#ax.set_ylabel('Y', fontsize=12)
#ax.set_zlabel('Marker Depth', fontsize=12)
#ax.tick_params(labelsize=6)
#plt.title('Simple Kriging')
#plt.show()
#fig.savefig('Fig/SK_3D.png', dpi=600)

##########################################################################



        
        
