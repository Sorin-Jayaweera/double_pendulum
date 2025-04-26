#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 14:35:43 2025

@author: alyrajan

@author: wunderbear
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy.solvers.ode as ode
import sympy as sym
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time


def matStuff(theta1, w1, theta2, w2, m1, l1, m2, l2, g):
    
    al = m1*l1**2 + m2*l2**2
    be = m2*l1*l2*np.cos(theta1 - theta2)
    de = m2*l1*l2*np.cos(theta1-theta2)
    ep = m2*l2**2
    
    A = np.matrix([(al, be), (de, ep)])
    
    gamma = m1*l1*l2*np.sin(theta1-theta2) * (w1-w2) * w2 + g*(m1+m2)*l1*np.sin(theta1) - m2*w1*w2*l1*l2*np.sin(theta1-theta2)

    #gamma = m2*w2*l1*l2*np.sin(theta1-theta2)*(w1-w2) - m2*w1*w2*l1*l2*np.sin(theta1 -theta2) +g*(m1+m2)*l1*np.sin(theta1) #m1*l1*l2*w2*np.sin(theta1 - theta2)*(w1-w2) + g*(m1 + m2)*l1*np.sin(theta1) - m2*w1*w2*l1*l2*np.sin(theta1 - theta2)
    phi = m2*w1*l1*l2*np.sin(theta1-theta2)*(w1-w2) + g*m2*l2*np.sin(theta2) + m2*w1*w2*l1*l2*np.sin(theta1-theta2) #(m2*l1*l2*np.sin(theta1 - theta2)*(w1-w2))*w1 + g*m2*l2*np.sin(theta2) + m2*theta2*l2**2 + m2
    v = np.matrix([[gamma], [phi]])
    
    invA = np.matrix.getI(A)
    values = invA * v

    #print("inv: ", invA)
    #print("v: ", v)
    #print("values: ", values)

  
    return values[0],values[1]
    

def derivatives(t, X, m1, l1, m2, l2, g):
    theta1, w1, theta2, w2 = X
    #derivs = np.array([w1,theta1 ,w2,theta2]) #TODO: PUT IN THE EQUATIONS
    res = matStuff(theta1, w1, theta2, w2, m1, l1, m2, l2, g)
    
    
    a1 = res[0].item(0)
    a2 = res[1].item(0)
    
    return [w1, a1, w2, a2]

def energy(theta1, w1, theta2, w2, m1, m2, l1, l2, g):
    print("----- energy")
    print("theta1: ", theta1, ", theta2 ", theta2, " w1 ", w1, ", w2 ", w2)
    T = 1/2*m1*(w1**2)*(l1**2) + 1/2*m2*((w2**2) * (l2**2) + w1**2 * l2**2 + 2*abs(w1)*abs(w2)*l1*l2*np.cos(theta1 - theta2))
    #U = abs(g)*(  abs((m1+m2) *l1*np.cos(theta1)) + abs(m2*l2*np.cos(theta2) ))
    h = l1 + l2
    y1 = h-(l1*np.cos(theta1))

    y2 = y1 - l2*np.cos(theta2)
    
    U= abs(y1*m1*g) + abs(y2*m2*g)

    print("here: ", np.cos(theta1-theta2))
    return T, U, T+U

energyVec = np.vectorize(energy)

def graphEnergy(timeArr, theta1arr, w1arr, theta2arr, w2arr, m1, m2, l1, l2, g):
    print("----- graph energy")
    idx = 0
    print("theta1: ", theta1arr[idx], ", theta2 ", theta2arr[idx], " w1 ", w1arr[idx], ", w2 ", w2arr[idx])
   
    if np.size(timeArr) != np.size(theta1arr) != np.size(theta2arr) != np.size(w1arr) != np.size(w2arr):
        raise Exception("Sorry, no numbers below zero")
    
    m1Vec = np.full(np.size(theta1arr), m1)
    m2Vec = np.full(np.size(theta1arr), m2)
    l1Vec = np.full(np.size(theta1arr), l1)
    l2Vec = np.full(np.size(theta1arr), l2)
    gVec = np.full(np.size(theta1arr), g)
    tVect, uVec, totVec = energyVec(theta1arr, w1arr, theta2arr, w2arr, m1Vec, m2Vec, l1Vec, l2Vec, gVec)
    
    fig, ax = plt.subplots()

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    
    #plt.plot(timeArr, tVect, "ro", label="Kinetic Energy")
    #plt.plot(timeArr, uVec, "bo", label="Potential Energy")
    plt.plot(timeArr, totVec, "ro", label="Total Energy")
    plt.set_xlim([])
    plt.legend()
    plt.show()
    



upperlim = 50
numpoints = 40 * upperlim +1

m1,l1,m2,l2, g = [1,1,1,1,-10]#-9.8]

theta10 = np.pi/4#np.pi/3
theta20 = np.pi/4

res1 = solve_ivp(derivatives, [0, upperlim], [theta10,0,theta20,0],t_eval=np.linspace(0,upperlim,numpoints),args=(m1,l1,m2,l2,g), rtol= 2.2205e-14, atol=1e-30)



plt.close("all")


#graphEnergy(timeArr, theta1arr, w1arr, theta2arr, w2arr, m1, m2, l1, l2, g):

graphEnergy(np.linspace(0,upperlim,numpoints), res1.y[0,:], res1.y[1,:], res1.y[2,:], res1.y[3,:], m1, m2, l1, l2, g)


theta1 = res1.y[0,:]
theta2 = res1.y[2,:]
t1 = res1.t
x1 = l1 * np.sin(theta1)

h = l1 + l2

y1 = h-(l1*np.cos(theta1))

x2 = x1 + l2* np.sin(theta2)
y2 = y1 - l2*np.cos(theta2)



fig, ax = plt.subplots()

ax.set_xlabel('T [Samples]')
ax.set_ylabel('X')

ax.set_aspect(1)
ax.scatter(0,l1)
ax.scatter(0,l2)





def update(i):
    ax.clear()

    ax.set_aspect(1)
    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    plt.xlim(-l1-l2, l1+l2)
    plt.ylim(0, 2*(l1+l2))
    
    ax.scatter(x1[i],y1[i])
    ax.plot([0,x1[i]],[h,y1[i]])
    ax.scatter(x2[i],y2[i])
    ax.plot([x1[i],x2[i]],[y1[i],y2[i]])

    
ani = FuncAnimation(fig=fig, func=update, frames=len(x1), interval=30)
plt.show()

