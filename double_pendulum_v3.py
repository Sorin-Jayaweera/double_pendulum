#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 20:43:15 2025

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

from IPython.display import HTML

def matStuff(theta1, w1, theta2, w2, m1, l1, m2, l2, g):
    
    al = m1*l1**2 + m2*l2**2
    be = m2*l1*l2*np.cos(theta1 - theta2)
    de = m2*l1*l2*np.cos(theta1-theta2)
    ep = m2*l2**2
    
    A = np.matrix([(al, be), (de, ep)])
    
    gamma = m1*l1*l2*np.sin(theta1-theta2) * (w1-w2) * w2 + g*(m1+m2)*l1*np.sin(theta1) - m2*w1*w2*l1*l2*np.sin(theta1-theta2)
    phi = m2*w1*l1*l2*np.sin(theta1-theta2)*(w1-w2) + g*m2*l2*np.sin(theta2) + m2*w1*w2*l1*l2*np.sin(theta1-theta2) #(m2*l1*l2*np.sin(theta1 - theta2)*(w1-w2))*w1 + g*m2*l2*np.sin(theta2) + m2*theta2*l2**2 + m2
    v = np.matrix([[gamma], [phi]])
    
    invA = np.matrix.getI(A)
    values = invA * v

    print("inv: ", invA)
    print("v: ", v)
    print("values: ", values)

  
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
    ax.plot(timeArr, totVec, "ro", label="Total Energy")
    ax.set_xlim([])
    ax.legend()
    ax.show()
    
# initialization
upperlim = 50#seconds
numpoints = 20 * upperlim +1 # pps * seconds + 1 for nice number
numPendulums = 20
m1,l1,m2,l2, g = [1,1,1,1,-9.8]


epsilon = 0.001

m1arr = np.full(numPendulums,m1)
l1arr =  np.full(numPendulums,l1)
m2arr =  np.full(numPendulums,m2)
l2arr =  np.full(numPendulums,l2)
garr =  np.full(numPendulums,g)


theta10 = np.pi/2
theta10arr = np.array([])
for i in range(numPendulums):
    theta10arr = np.append(theta10arr,theta10+i*epsilon)

theta20 = np.pi
theta20arr = np.array([])
for i in range(numPendulums):
    theta20arr = np.append(theta20arr,theta20+i*epsilon)
    
t = np.linspace(0,upperlim,numpoints)
resArr = np.array([])
for i in range(len(m1arr)):
    rtol = 1e-5#1e-7#2.2205e-14
    atol = 1e-7#1e-9#1e-30
    resArr = np.append(resArr, solve_ivp(derivatives, [0, upperlim], [theta10arr[i],0,theta20arr[i],0],t_eval=t,args=(m1arr[i],l1arr[i],m2arr[i],l2arr[i],garr[i]),rtol=rtol,atol=atol))
    
    
## Distance metric


optheta1 = resArr[0].y[0,:] # Origional pendulum phase
optheta2 = resArr[0].y[2,:]
distArr = np.array([optheta1**2+optheta2**2]) #reference distance

for i in range(1,numPendulums):

    theta1 = resArr[i].y[0,:]
    theta2 = resArr[i].y[2,:]
    distArr = np.append(distArr, [theta1 * optheta1 + theta2 * optheta2],axis=0) # dot product

## Distance metric

optheta1 = resArr[0].y[0,:] # Origional pendulum phase
optheta2 = resArr[0].y[2,:]
distArr = np.array([0*optheta1]) #reference distance

for i in range(1,numPendulums):

    theta1 = resArr[i].y[0,:]
    theta2 = resArr[i].y[2,:]
    distArr = np.append(distArr, [((theta1 - optheta1 )**2 + (theta2 - optheta2)**2)**(1/2)],axis=0) #quadrature with the first pendulum
"""
## Distance metric

optheta1 = resArr[0].y[0,:] # Origional pendulum phase
optheta2 = resArr[0].y[2,:]
lasttheta1 = optheta1
lasttheta2 = optheta2
distArr = np.array([0*optheta1]) #reference distance

for i in range(1,numPendulums):

    theta1 = resArr[i].y[0,:]
    theta2 = resArr[i].y[2,:]
    distArr = np.append(distArr, [((theta1 - lasttheta1 )**2 + (theta2 - lasttheta2)**2)**(1/2)],axis=0) #quadrature with the preceeding pendulum
    lasttheta1 = theta1
    lasttheta2 = theta2
plt.rcParams['animation.embed_limit'] = 2**128

"""
plt.close("all")


fig, (ax1,ax2,ax3) = plt.subplots(1,3)


for i in range(numPendulums):
        
    theta1 = resArr[i].y[0,:]
    theta2 =  resArr[i].y[2,:]
    
    w1 = resArr[i].y[1,:]
    w2 =  resArr[i].y[3,:]
    t1 =  resArr[i].t
    x1 = l1 * np.sin(theta1)

    h = l1 + l2

    y1 = h-(l1*np.cos(theta1))

    x2 = x1 + l2* np.sin(theta2)
    y2 = y1 - l2*np.cos(theta2)

    T = 1/2*m1*(w1**2)*(l1**2) + 1/2*m2*((w2**2) * (l2**2) + w1**2 * l2**2 + 2*abs(w1)*abs(w2)*l1*l2*np.cos(theta1 - theta2))
    U = abs(y1*m1*g) + abs(y2*m2*g)
   
   
    h = l1 + l2
    y1 = h-(l1*np.cos(theta1))


    y2 = y1 - l2*np.cos(theta2)
    
    ax1.scatter(0,l1arr[i])
    ax1.scatter(0,l2arr[i])

    ax2.scatter(t, distArr[i])

    ax3.scatter(t,U)
    
    


def update(frame):
    ax1.clear()

    ax2.clear()
    
    for i in range(numPendulums-1,-1,-1):
            
        theta1 = resArr[i].y[0,:]
        theta2 =  resArr[i].y[2,:]
        t1 =  resArr[i].t
        x1 = l1arr[i] * np.sin(theta1)

        h = l1arr[i] + l2arr[i]

        y1 = h-(l1arr[i]*np.cos(theta1))

        x2 = x1 + l2arr[i]* np.sin(theta2)
        y2 = y1 - l2arr[i]*np.cos(theta2)
        
        if(i == 0):
                
            ax1.scatter(x1[frame],y1[frame]) # point 1
            ax1.plot([0,x1[frame]],[h,y1[frame]],linewidth=4) # line from 0 - 1
            ax1.scatter(x2[frame],y2[frame]) # point 2
            ax1.plot([x1[frame],x2[frame]],[y1[frame],y2[frame]],linewidth=4) # line from 1 - 2
        else:
                
            ax1.scatter(x1[frame],y1[frame]) # point 1
            ax1.plot([0,x1[frame]],[h,y1[frame]],linewidth=1) # line from 0 - 1
            ax1.scatter(x2[frame],y2[frame]) # point 2
            ax1.plot([x1[frame],x2[frame]],[y1[frame],y2[frame]],linewidth=1) # line from 1 - 2
                
        ax1.set_xlim(-l1arr[i]-l2arr[i], l1arr[i]+l2arr[i])
        ax1.set_ylim(0, 2*(l1arr[i]+l2arr[i]))
        ax1.set_xlabel('T [Samples]')
        ax1.set_ylabel('X')

        ax1.set_aspect(1)

    for i in range(numPendulums-1,0,-1):
        ax2.scatter(t[1:frame], distArr[i][1:frame])
        ax2.set_title("Distance Metric")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Distance")
    
    fig.tight_layout()
ani = FuncAnimation(fig=fig, func=update, frames= len(t), interval=10) 

plt.show()

HTML(ani.to_jshtml())

maxw1 = 0
maxw2 = 0
maxtheta1 = 0
maxtheta2 = 0
minw1 = 0
minw2 = 0
mintheta1 = 0
mintheta2 = 0
twopi = 2*np.pi
for i in range(numPendulums):
    theta1 = resArr[i].y[0,:]
    theta2 =  resArr[i].y[2,:]
    w1 = resArr[i].y[1,:]
    w2 = resArr[i].y[3,:]
    if(max(w1) > maxw1):
        maxw1 = max(w1)

    if(max(w2) > maxw2):
        maxw2 = max(w2)
        
    if(max(theta1) > maxtheta1):
        maxtheta1 = max(theta1)
    if(max(theta2) > maxtheta2):
        maxtheta2 = max(theta2)

    if(min(w1) < minw1):
        minw1 =  min(w1)

    if(min(w2) < minw2):
        minw2 = min(w2)
        
    if(min(theta1) < mintheta1):
        mintheta1 = min(theta1)
    if(min(theta2) < mintheta2):
        mintheta2 = min(theta2)

fig, axs = plt.subplots(2,2)

def update(frame):
    axs[0, 0].clear()
    axs[0, 1].clear()
    axs[1,0 ].clear()
    axs[1,1].clear()

    for i in range(len(m1arr)):
        
        theta1 = resArr[i].y[0,:]
        theta2 =  resArr[i].y[2,:]
        w1 = resArr[i].y[1,:]
        w2 = resArr[i].y[3,:]
        
        axs[0, 0].scatter(theta1[frame],theta2[frame])#% twopi
        axs[0, 1].scatter(theta1[frame], w1[frame])
        axs[1,0].scatter(theta2[frame], w2[frame])
        axs[1,1].scatter(w1[frame], w1[frame])

    axs[0, 0].set_xlim([mintheta1, maxtheta1])#[-2*np.pi,2*np.pi])
    axs[0, 0].set_ylim([mintheta2, maxtheta2])#[-2*np.pi,2*np.pi])
    axs[0, 0].set_xlabel('Theta 1')
    axs[0, 0].set_ylabel('Theta 2')
    
    
    axs[0, 1].set_xlim([mintheta1,maxtheta1])
    axs[0, 1].set_ylim([minw1,maxw1])
    axs[0, 1].set_xlabel('Theta 1')
    axs[0, 1].set_ylabel('Omega 1')
    
    
    axs[1,0].set_xlim([mintheta2,maxtheta2])
    axs[1,0].set_ylim([minw2,maxw2])
    axs[1,0].set_xlabel('Theta 2')
    axs[1,0].set_ylabel('Omega 2')
    
    
    axs[1,1].set_xlim([minw1,maxw1])
    axs[1,1].set_ylim([minw2,maxw2])
    axs[1,1].set_xlabel('Omega 1')
    axs[1,1].set_ylabel('Omega 2')

    fig.suptitle("Phase Space of Pendulums")
    #fig.tight_layout()
ani = FuncAnimation(fig=fig, func=update, frames=len(t), interval=30)
