# NoahPy
A new version of the backpropagation support for the land surface model

## Main Process

![NoahLSM](https://github.com/user-attachments/assets/2d062cde-37a6-49e0-ad69-8dd7b2564081)
from https://ral.ucar.edu/model/unified-noah-lsm

### Heat equation
$$\frac{\partial}{\partial t }\rho C_p T=\frac{\partial}{\partial z}[\frac{\partial K T}{\partial z}]+Q$$

### Richards equation 

$$\frac{\partial \theta}{\partial t} =\frac{\partial }{\partial z}[D(\theta)\frac{\partial \theta}{\partial z}] +\frac{\partial K(\theta)}{\partial z}+S$$

Discretize using a differential format
$$\frac{\theta_{k}^{n+1}-\theta_{k}^{n}}{\Delta t} = \frac{1}{\Delta z_k}[D(\theta_{k-1})\frac{\theta_{k-1}^{n+1}-\theta_{k}^{n+1}}{\Delta \tilde{z}_{k-1}}-  D(\theta_{k})\frac{\theta_{k}^{n+1}-\theta_{k+1}^{n+1}{\Delta \tilde{z}_{k}} + S]$$

