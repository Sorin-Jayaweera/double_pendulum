#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:14:07 2025

@author: alyrajan
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
import math as m
import pandas as pd

def matStuff(theta1, w1, theta2, w2, m1, l1, m2, l2, g):
    """ 
    Given the values for theta1, w1, theta2, w2, m1, l1, m2, l2 and g
    Returns the value for α1 and α2
    """
    
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

  
    return values[0],values[1]
    
# why do we need t for this? Can we remove it?
def derivatives(t, X, m1, l1, m2, l2, g):
    theta1, w1, theta2, w2 = X
    res = matStuff(theta1, w1, theta2, w2, m1, l1, m2, l2, g)
    
    a1 = res[0].item(0)
    a2 = res[1].item(0)
    
    return [w1, a1, w2, a2]

def energy(theta1, w1, theta2, w2, m1, m2, l1, l2, g):
    """ 
    Given the values for theta1, w1, theta2, w2, m1, m2, l1, l2 and g
    Returns the energy of the system in the form of kinetic, potential, total
    """
    T = 1/2*m1*(w1**2)*(l1**2) + 1/2*m2*((w2**2) * (l2**2) + w1**2 * l2**2 + 2*abs(w1)*abs(w2)*l1*l2*np.cos(theta1 - theta2))
    #U = abs(g)*(  abs((m1+m2) *l1*np.cos(theta1)) + abs(m2*l2*np.cos(theta2) ))
    h = l1 + l2
    y1 = h-(l1*np.cos(theta1))

    y2 = y1 - l2*np.cos(theta2)
    
    U= abs(y1*m1*g) + abs(y2*m2*g)

    return T, U, T+U

energyVec = np.vectorize(energy)

def graphEnergy(timeArr, theta1arr, w1arr, theta2arr, w2arr, m1, m2, l1, l2, g):
   
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
upperlim = 10#seconds
numpoints = 40 * upperlim # pps * seconds + 1 for nice number
numPendulums = 2
m1,l1,m2,l2, g = [1,1,1,1,-9.8]

epsilon = 0.01

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

"""
def distanceMetric(theta1, optheta1, theta2, optheta2, w1, opw1, w2, opw2):
    return ((theta1 - optheta1 )**2 + (theta2 - optheta2)**2 + (w1-opw1)**2 + (w2-opw2)**2)**(1/2)
"""

def simulatorHelper(m1_1, l1_1, m2_1, l2_1, theta10_1, theta20_1,
              m1_2, l1_2, m2_2, l2_2, theta10_2, theta20_2,
              g, upperlim, rtol=1e-9, atol=1e-9, epsilon=0.01):

    pps = 40
    numpoints = pps * upperlim
    t_eval = np.linspace(0, upperlim, numpoints)
    
    # Solve for Pendulum 1
    sol1 = solve_ivp(
        derivatives,
        [0, upperlim],
        [theta10_1, 0, theta20_1, 0],
        t_eval=t_eval,
        args=(m1_1, l1_1, m2_1, l2_1, g),
        rtol=rtol,
        atol=atol
    )

    # Solve for Pendulum 2
    sol2 = solve_ivp(
        derivatives,
        [0, upperlim],
        [theta10_2, 0, theta20_2, 0],
        t_eval=t_eval,
        args=(m1_2, l1_2, m2_2, l2_2, g),
        rtol=rtol,
        atol=atol
    )

    # Assemble DataFrame
    df = pd.DataFrame({
        "time": sol1.t,
        "theta1_1": sol1.y[0],
        "omega1_1": sol1.y[1],
        "theta2_1": sol1.y[2],
        "omega2_1": sol1.y[3],
        "theta1_2": sol2.y[0],
        "omega1_2": sol2.y[1],
        "theta2_2": sol2.y[2],
        "omega2_2": sol2.y[3]
    })
    df = df[[
        "time",
        "theta1_1", "theta2_1", "omega1_1", "omega2_1",
        "theta1_2", "theta2_2", "omega1_2", "omega2_2"
    ]]
    
    """
    df["Distance Metric"] = ((df["theta1_2"] - df["theta1_1"] )**2 + (df["theta2_2"] - df["theta2_1"])**2 + (df["omega1_2"]-df["omega1_1"])**2 + (df["omega2_2"]-df["omega2_1"])**2)**(1/2)
    
    plt.plot(df["time"], df["Distance Metric"], "bo")
    plt.show()
    """
    return df


def distanceMetric(df, epsilon):
    df["Distance Metric"] = ((df["theta1_2"] - df["theta1_1"] )**2 + (df["theta2_2"] - df["theta2_1"])**2 + (df["omega1_2"]-df["omega1_1"])**2 + (df["omega2_2"]-df["omega2_1"])**2)**(1/2)
    
    df["lyap"] = (np.log(df["Distance Metric"]/epsilon))/df["time"]
    
    plt.plot(df["time"], df["lyap"], "bo")
    plt.title("Lyap against time")
    plt.show()
    
    return df
    
    
    
    

def simulator(m1_1, l1_1, m2_1, l2_1, theta10_1, theta20_1,
              m1_2, l1_2, m2_2, l2_2, theta10_2, theta20_2,
              g, upperlim, rtol=1e-9, atol=1e-9, epsilon=0.5):
    
    df = simulatorHelper(
        m1_1, l1_1, m2_1, l2_1, theta10_1, theta20_1,
        m1_2, l1_2, m2_2, l2_2, theta10_2, theta20_2,
        g, upperlim)
    
    df = distanceMetric(df, epsilon)
    
    #plt.plot(df["time"], df["Distance Metric"], "ro")
    #plt.show()
    return df


def massGrapher():
    epsilon = 0.5
    m1_1 = 1.0
    m1_2 = 0.01
    m2_1 = 1.0
    m2_2 = 0.01
    data = pd.DataFrame({"Mass": [], "Time": []})
    for i in range(5):
        m1_2 += epsilon
        m2_2 += epsilon
        if m1_1 == m1_2 and m2_1 == m2_2:
            pass
        
        found = False
        upperlim = 10
        while found == False:
            df = simulator(
                m1_1=m1_1, l1_1=1.0, m2_1=m2_1, l2_1=1.0, theta10_1=np.pi/2, theta20_1=np.pi,
                m1_2=m1_2, l1_2=1.0, m2_2=m2_1, l2_2=1.0, theta10_2=np.pi/2 + 0.01, theta20_2=np.pi + 0.01,
                g=-9.8, upperlim=upperlim, epsilon=epsilon
            )
            
            print(df)
            distances = df["lyap"].to_list()
            
            
            for n in distances:
                if n >=1:
                    past = df[df['lyap'] == n]
                    first_row_dict = past.iloc[0].to_dict()
                    data.loc[-1] = [m1_2, first_row_dict["time"]]
                    found = True
                    
            upperlim*=2
    """
    print(data)
    plt.plot(data["Mass"], data["Time"], "ro")
    plt.xscale('log')
    plt.show()
    """          
                    
                    

"""
df = simulatorHelper(
    m1_1=1.0, l1_1=1.0, m2_1=1.0, l2_1=1.0, theta10_1=np.pi/2, theta20_1=np.pi,
    m1_2=1.0, l1_2=1.0, m2_2=1.0, l2_2=1.0, theta10_2=np.pi/2+0.01, theta20_2=np.pi+0.01,
    g=-9.8, upperlim=upperlim
)    
""" 
massGrapher()     
    
    



## Distance metric

optheta1 = resArr[0].y[0,:] # Origional pendulum phase
optheta2 = resArr[0].y[2,:]
opw1 = resArr[0].y[1,:] 
opw2 = resArr[0].y[3,:]

distArr = np.array([0*optheta1]) #reference distance

for i in range(1,numPendulums):

    theta1 = resArr[i].y[0,:]
    theta2 = resArr[i].y[2,:]
    w1 = resArr[i].y[1,:]
    w2 = resArr[i].y[3,:]
    distArr = np.append(distArr, [((theta1 - optheta1 )**2 + (theta2 - optheta2)**2 + (w1-opw1)**2 + (w2-opw2)**2)**(1/2)],axis=0) #quadrature with the first pendulum


Etot = np.array([0*t])
Uarr = np.array([0*t])
Tarr = np.array([0*t])

for i in range(numPendulums):
    theta1 = resArr[i].y[0,:]
    theta2 =  resArr[i].y[2,:]
    
    w1 = resArr[i].y[1,:]
    w2 =  resArr[i].y[3,:]
    t1 =  resArr[i].t

    h = l1 + l2

    x1 = l1 * np.sin(theta1)
    y1 = h-(l1*np.cos(theta1))

    x2 = x1 + l2* np.sin(theta2)
    y2 = y1 - l2*np.cos(theta2)

    T = 1/2*m1*(w1**2)*(l1**2) + 1/2*m2*((w2**2) * (l2**2) + w1**2 * l2**2 + 2*abs(w1)*abs(w2)*l1*l2*np.cos(theta1 - theta2))
    U = abs(y1*m1*g) + abs(y2*m2*g)
   
    Etot = np.append(Etot, [U+T],axis=0)

    Uarr = np.append(Uarr, [U],axis=0)
    Tarr = np.append(Tarr, [T],axis=0)

Etot = Etot[1:,:]
Tarr = Tarr[1:,:]
Uarr = Uarr[1:,:]

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
        
plt.rcParams['animation.embed_limit'] = 2**128
plt.rcParams["figure.figsize"] = (7,7)

plt.close("all")

fig, ax = plt.subplots(4,2)


def update(frame):
    ax[0,0].clear()
    ax[0,1].clear()
    ax[1,0].clear()
    ax[1,1].clear()

    ax[2,0].clear()
    ax[2,1].clear()
    ax[3,0 ].clear()
    ax[3,1].clear()    
    
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
                
            ax[0,0].scatter(x1[frame],y1[frame]) # point 1
            ax[0,0].plot([0,x1[frame]],[h,y1[frame]],linewidth=4) # line from 0 - 1
            ax[0,0].scatter(x2[frame],y2[frame]) # point 2
            ax[0,0].plot([x1[frame],x2[frame]],[y1[frame],y2[frame]],linewidth=4) # line from 1 - 2
        else:
                
            ax[0,0].scatter(x1[frame],y1[frame]) # point 1
            ax[0,0].plot([0,x1[frame]],[h,y1[frame]],linewidth=1) # line from 0 - 1
            ax[0,0].scatter(x2[frame],y2[frame]) # point 2
            ax[0,0].plot([x1[frame],x2[frame]],[y1[frame],y2[frame]],linewidth=1) # line from 1 - 2
                
        ax[0,0].set_xlim(-l1arr[i]-l2arr[i], l1arr[i]+l2arr[i])
        ax[0,0].set_ylim(0, 2*(l1arr[i]+l2arr[i]))
        ax[0,0].set_xlabel('T [Samples]')
        ax[0,0].set_ylabel('X')

        ax[0,0].set_aspect(1)

    for i in range(numPendulums-1,0,-1):
        ax[0,1].scatter(t1[1:frame], distArr[i][1:frame])
        ax[0,1].set_title("Distance Metric")
        ax[0,1].set_xlabel("Time")
        ax[0,1].set_ylabel("Distance")     

        ax[1,0].scatter(t1[1:frame],Etot[i][1:frame])    
        ax[1,0].set_title("Energy Conservation")
        ax[1,0].set_xlabel("Time")
        ax[1,0].set_ylabel("Totatl Energy")     
        
        ax[1,1].plot(t1[1:frame],Uarr[i][1:frame],label="Potential")     
        ax[1,1].plot(t1[1:frame],Tarr[i][1:frame],label="Kinetic")    
        ax[1,1].set_title("Energy Breakdown")
        ax[1,1].set_xlabel("Time")
        ax[1,1].set_ylabel("Energy")        
    
    for i in range(numPendulums):
        
        theta1 = resArr[i].y[0,:]
        theta2 =  resArr[i].y[2,:]
        w1 = resArr[i].y[1,:]
        w2 = resArr[i].y[3,:]
        
        ax[2, 0].scatter(theta1[frame],theta2[frame])#% twopi
        ax[2, 1].scatter(theta1[frame], w1[frame])
        ax[3,0].scatter(theta2[frame], w2[frame])
        ax[3,1].scatter(w1[frame], w1[frame])

        ax[2, 0].set_xlim([mintheta1, maxtheta1])#[-2*np.pi,2*np.pi])
        ax[2, 0].set_ylim([mintheta2, maxtheta2])#[-2*np.pi,2*np.pi])
        ax[2, 0].set_xlabel('Theta 1')
        ax[2, 0].set_ylabel('Theta 2')
        
        
        ax[2, 1].set_xlim([mintheta1,maxtheta1])
        ax[2, 1].set_ylim([minw1,maxw1])
        ax[2, 1].set_xlabel('Theta 1')
        ax[2, 1].set_ylabel('Omega 1')
                
        ax[3,0].set_xlim([mintheta2,maxtheta2])
        ax[3,0].set_ylim([minw2,maxw2])
        ax[3,0].set_xlabel('Theta 2')
        ax[3,0].set_ylabel('Omega 2')
        
        
        ax[3,1].set_xlim([minw1,maxw1])
        ax[3,1].set_ylim([minw2,maxw2])
        ax[3,1].set_xlabel('Omega 1')
        ax[3,1].set_ylabel('Omega 2')
    fig.tight_layout()
ani = FuncAnimation(fig=fig, func=update, frames=len(t) , interval=10) 

plt.show()

