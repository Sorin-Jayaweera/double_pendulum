## Double pendulum

The Lagrangian $\mathscr{L}$ is given by
$$
\begin{align}
\mathscr{L}  & =  T-U \\
\end{align}
$$

$$
\begin{align}
T = \frac{1}{2}m_{1} \dot{\theta}_{1}^{2}l_{1}^{2} + \frac{1}{2} m_{2} (\dot{\theta}^{2}_{2}l_{2}^{2} + \dot{\theta}_{1}^{2}l_{2}^{2}+2 \dot{\theta_{1}}\dot{\theta_{2}}l_{1}l_{2}\cos(\theta_{1}-\theta_{2}) ) \\
U = g\bigg[ (m_{1}+m_{2})l_{1}\cos (\theta_{1}) + m_{2}l_{2}\cos\theta_{2} \bigg] 
\end{align}
$$
so
$$
\begin{align}
\mathscr{L}  = \frac{1}{2}m_{1} \dot{\theta}_{1}^{2}l_{1}^{2} + \frac{1}{2} m_{2} (\dot{\theta}^{2}_{2}l_{2}^{2} + \dot{\theta}_{1}^{2}l_{2}^{2}+2 \dot{\theta_{1}}\dot{\theta_{2}}l_{1}l_{2}\cos(\theta_{1}-\theta_{2}) ) -g\bigg[ (m_{1}+m_{2})l_{1}\cos (\theta_{1}) + m_{2}l_{2}\cos\theta_{2} \bigg] 
\end{align}
$$

We have the differential equation
$$
\begin{align}
\frac{ d }{d t } \frac{ \partial \mathscr{L}  }{ \dot{\partial}q }  = \frac{ \partial L }{ \partial q } 
\end{align}
$$
We take the derivatives with $\theta_{1},\theta_{2}, \dot{\theta_{1}},\dot{\theta_{2}}$

First, $\theta_{1}$. Lets take the two derivatives:
$$
\begin{align}
\frac{ \partial \mathscr{l}  }{ \partial \theta_{1} }  & = g(m_{1}+m_{2}) l_{1} \sin(\theta_{1}) - m_{2} \dot{\theta_{1}} \dot{\theta_{2}} l_{1}l_{2} \sin(\theta_{1}-\theta_{2})  \\
\frac{ \partial \mathscr{l} }{ \partial \dot{\theta}_{1} }  & = l_{1}^{2}m_{1} \dot{\theta_{1}} + m_{2}l_{2}^{2} \dot{\theta_{1}} + m_{2} \dot{\theta_{2}}l_{1}l_{2} \cos(\theta_{1}-\theta_{2})  \\
\end{align}
$$

$$
\begin{align}
\frac{ d }{d t } \frac{ \partial \mathscr{l}  }{ \partial \dot{\theta}_{1}  } = (m_{1}l_{1}^{2}+m_{2}l_{2}^{2} ) \ddot{\theta}_{1}  + m_{2}l_{1}l_{2} (\ddot{\theta}_{2} \cos(\theta_{1}-\theta_{2}) ) - m_{1}l_{1}l_{2}\sin(\theta_{1}-\theta_{2})(\dot{\theta_{1}}-\dot{\theta_{2}}) \dot{\theta}_{2} 
\end{align}
$$


We have the LaGrange equation of motion for $\theta_{1}$ (rearranging to have the accelerations on the left and others on the right):
$$
\begin{align}
(m_{1}l_{1}^{2}+m_{2}l_{2}^{2})\ddot{\theta_{1}} + m_{2}l_{1}l_{2} \cos(\theta_{1}-\theta_{2}) \ddot{\theta}_{2} = m_{1}l_{1}l_{2}\sin(\theta_{1}-\theta_{2})(\dot{\theta_{1}}-\dot{\theta}_{2}) \dot{\theta_{2}} + g(m_{1}+m_{2})l_{1}\sin(\theta)_{1} - m_{2} \dot{\theta}_{1}\dot{\theta}_{2}l_{1}l_{2} \sin(\theta_{1}-\theta_{2})
\end{align}
$$


We also have the derivatives with $\theta_{2}$:

$$
\begin{align}
\frac{ \partial \mathscr{L}  }{ \partial \theta_{2} } = gm_{2}l_{2} \sin(\theta_{2}) + m_{2} \dot{\theta}_{1} \dot{\theta}_{2 } l_{1}l_{2} \sin(\theta_{1}-\theta_{2}) \\
\frac{ \partial \mathscr{L}  }{ \partial \dot{\theta}_{2}  }  = m_{2} \dot{\theta}_{2} l_{2}^{2} + m_{2} \dot{\theta_{1}}l_{1}l_{2} \cos(\theta_{1}-\theta_{2}) 
\end{align}
$$
$$
\begin{align}
\frac{ d }{d t } \frac{ \partial \mathscr{L}  }{ \dot{\partial}\theta_{2} } = m_{2}l_{2}^{2} \ddot{\theta}_{2} + m_{2}l_{1}l_{2} \cos(\theta_{1}-\theta_{2})\ddot{\theta}_{1} - m_{2} \dot{\theta}_{1} l_{1}l_{2}\sin(\theta_{1}-\theta_{2})(\dot{\theta}_{1} - \dot{\theta}_{2} )
\end{align}
$$

Which gets the equation of motion

For $\theta_{2}$:
$$
\begin{align}
m_{2}l_{1}l_{2} \cos(\theta_{1}-\theta_{2})\ddot{\theta}_{1}  +m_{2}l_{2}^{2} \ddot{\theta}_{2}  = m_{2} \dot{\theta}_{1}l_{1}l_{2}\sin(\theta_{1}-\theta_{2})(\dot{\theta_{1}-\dot{\theta_{2}}}) 
\end{align}
$$


We need to solve the equations
$$
\begin{align}
\alpha \ddot{\theta}_{1} + \beta \ddot{\theta}_{2} = \gamma \\
\delta \ddot{\theta}_{1} + \epsilon \ddot{\theta}_{2} = \phi
\end{align}
$$
So we have the matrix operation
$$
\begin{align}
\begin{bmatrix}
\alpha & \beta \\
\delta  &  \epsilon
\end{bmatrix} \begin{bmatrix}

\ddot{\theta_{1}}\\ \ddot{\theta_{2}}
\end{bmatrix} = \begin{pmatrix}
\gamma\\ \varphi 
\end{pmatrix}
\end{align}
$$




## Single Pendulum



The Lagrangian $\mathscr{L}$ is given by
$$
\begin{align}
\mathscr{L}  & =  T-U \\
\end{align}
$$

$$
\begin{align}
T = \frac{1}{2} m \omega^{2}\\
U = L(1-\cos\theta)mg
\end{align}
$$


so
$$
\begin{align}
\mathscr{L}  = \frac{1}{2} m\omega^{2} - Lm(1-\cos\theta)g
\end{align}
$$


First, Lets take the two derivatives:
$$
\begin{align}
\frac{ \partial \mathscr{l}  }{ \partial \theta_{1} }  & = -Lmg\sin\theta  \\
\frac{ \partial \mathscr{l}  }{ \partial \dot{\theta} }  & =  m\omega 
\end{align}
$$




We have the differential equation
$$
\begin{align}
\frac{ d }{d t } \frac{ \partial \mathscr{L}  }{ \dot{\partial}q }  = \frac{ \partial L }{ \partial q } 
\end{align}
$$

We now have the Lagrange equation of motion:
$$
\begin{align}
m \ddot{\theta} = -Lmg\sin\theta \\
\ddot{\theta} = -Lg\sin\theta
\end{align}
$$

