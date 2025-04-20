#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 14:35:43 2025

@author: alyrajan
"""

# IMPORTS
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy.solvers.ode as ode
import sympy as sym
import matplotlib.animation as animation

# Parameters of the system
theta = sym.symbols("θ")
dtheta = sym.symbols("dθ")
l ,m, g = [2,1,9.8]
L = sym.symbols("L")

U =  m * g * l * (1-sym.cos(theta)) 
T = 1/2 * m * dtheta**2
lagr = T-U 

ddtheta = sym.diff(lagr,theta)
dddtheta = sym.diff(lagr,dtheta)

#mddx = -glmsinx
# w= np.arcsin(-1*ddtheta/(g*l))
# dtheta = w
upperlim = 10
numpoints = 40 * upperlim +1

def derivatives(t, X, m, l, g):
    theta, w = X
    derivs = np.array([w, -l*g*np.sin(theta)])
    return derivs

theta0 = np.pi/2
res1 = solve_ivp(derivatives, [0, upperlim], [theta0,0],t_eval=np.linspace(0,upperlim,numpoints),args=(1,1,9.8))

plt.close("all")
t1 = res1.t
x = l * np.sin(res1.y[0,:])
y = l-(l * ( np.cos(res1.y[0,:])))

from matplotlib.animation import FuncAnimation
import time
        
    # linex = np.linspace(0,x[i],10)
    # liney = np.linspace(l,y[i],10)
fig, ax = plt.subplots()

ax.set_xlabel('T [Samples]')
ax.set_ylabel('X')

ax.set_aspect(1)
ax.scatter(0,l)

def update(i):
    ax.clear()
    ax.set_aspect(1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    ax.scatter(x[i],y[i])
    ax.plot([0,x[i]],[l,y[i]])
    #return (x[i], y[i])
    
ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=30)
plt.show()
