#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 14:35:43 2025

@author: alyrajan

@author: wunderbear
"""

# IMPORTS

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy.solvers.ode as ode
import sympy as sym
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time

# Parameters of the system
theta = sym.symbols("θ")
dtheta = sym.symbols("dθ")
m1,l1,m2,l2, g = [1,1,1,1,9.8]
L = sym.symbols("L")


upperlim = 20
numpoints = 40 * upperlim +1

def matStuff(m1, l1, m2, l2, theta1, w1, theta2, w2, g):
    
    al = m1*l1**2 + m2*l2**2
    be = m2*l1*l2*np.cos(theta1 - theta2)
    de = m2*l1*l2*np.cos(theta1-theta2)
    ep = m2*l2
    
    A = np.matrix((al, be), (de, ep))
    
    gamma = m1*l1*l2*w2*np.sin(theta1 - theta2)*(w1-w2) + g*(m1 + m2)*l1*np.sin(theta1) - m2*w1*w2*l1*l2*np.sin(theta1 - theta2)
    phi = (m2*l1*l2*np.sin(theta1 - theta2)*(w1-w2))*w1 + g*m2*l2*np.sin(theta2) + m2*theta2*l2**2 + m2
    v = np.matrix(gamma, phi)
    
    invA = np.matrix.getI(A)
    values = invA * v
    
    return values
    

def derivatives(t, X, m1, l1,m2,l2, g):
    theta1, w1, theta2, w2 = X
    derivs = np.array([w1,theta1 ,w2,theta2]) #TODO: PUT IN THE EQUATIONS

    return derivs

"""
theta10 = np.pi/2
theta20 = np.pi/2

res1 = solve_ivp(derivatives, [0, upperlim], [theta10,0,theta20,0],t_eval=np.linspace(0,upperlim,numpoints),args=(m1,l1,m2,l2,g))

plt.close("all")
t1 = res1.t
x = l * np.sin(res1.y[0,:])
y = l-(l * ( np.cos(res1.y[0,:])))

        
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

    
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(x), interval=30)
plt.show()
"""