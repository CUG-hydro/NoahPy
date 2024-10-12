# NoahPy
A new version of the backpropagation support for the land surface model

## Main Process

![NoahLSM](https://github.com/user-attachments/assets/2d062cde-37a6-49e0-ad69-8dd7b2564081)
from https://ral.ucar.edu/model/unified-noah-lsm

## Basic equation

In the land surface process model, the two most core basic laws are the heat equation and the water equation, both of which are partial differential equations. Finite difference methods are often used to solve these equations, that is, they are ultimately converted into a system of equations. Therefore, the differentiable solution method of machine learning platforms (such as Pytorch and TensorFlow) can be used to enable the model to propagate gradients.

### Heat equation
$$\frac{\partial}{\partial t }\rho C_p T=\frac{\partial}{\partial z}[\frac{\partial K T}{\partial z}]+Q$$

### Richards equation 

$$\frac{\partial \theta}{\partial t} =\frac{\partial }{\partial z}[D(\theta)\frac{\partial \theta}{\partial z}] +\frac{\partial K(\theta)}{\partial z}+S$$

---
### Finite Difference Discretization

#### Soil discrete schematic diagram
![Soil discrete schematic diagram](https://github.com/user-attachments/assets/cfb240a1-f0d3-4923-8242-e5cc1b93b072)
- Pan H L, Mahrt L. Interaction between soil hydrology and boundary-layer development[J]. Boundary-Layer Meteorology, 1987, 38(1-2): 185-202. [doi: 10.1007/BF00121563](https://link.springer.com/article/10.1007/BF00121563).
- Kalnay E, Kanamitsu M. Time Schemes for Strongly Nonlinear Damping Equations[J]. Monthly Weather Review, 1988, 116(10): 1945-1958. [doi: 10.1175/1520-0493(1988)116<1945:TSFSND>2.0.CO;2](https://journals.ametsoc.org/view/journals/mwre/116/10/1520-0493_1988_116_1945_tsfsnd_2_0_co_2.xml).

The finite difference of the equation can be obtained by using the above discretization scheme and the time scheme "D" (IMPLICIT STATE, EXPLICIT COEFFICIENT)  of section  2 of Kalnay AND Kanamitsu:

**We can get:**

```math
\begin{align}
\frac{\theta_{k}^{n+1}-\theta_{k}^{n}}{\Delta t} = \frac{1}{\Delta z_k}[D(\theta_{k-1})\frac{\theta_{k-1}^{n+1}-\theta_{k}^{n+1}}{\Delta \tilde{z}_{k-1}}-  D(\theta_{k})\frac{\theta_{k}^{n+1}-\theta_{k+1}^{n+1}}{\Delta \tilde{z}_{k}} + S] \\\\\
\frac{\theta_{k}^{n+1}-\theta_{k}^{n}}{\Delta t} =-\frac{D(\theta_{k-1})}{\Delta z_k \Delta \tilde{z}_{k-1}} (\theta_{k}^{n+1}-\theta_{k-1}^{n+1})-\frac{D(\theta_{k})}{\Delta z_k \Delta \tilde{z}_{k}}(\theta_{k}^{n+1}-\theta_{k+1}^{n+1})+\frac{S}{\Delta z_k}
\end{align}
```

**Let:**
```math
A = -\frac{D(\theta_{k-1})}{\Delta z_k \Delta \tilde{z}_{k-1}},C=-\frac{D(\theta_{k})}{\Delta z_k \Delta \tilde{z}_{k}}
```
**Finally:**
```math
\begin{align}
A\Delta t(\theta_{k-1}^{n+1}-\theta_{k-1}^{n})+B(\theta_{k}^{n+1}-\theta_{k}^{n})+C\Delta t(\theta_{k+1}^{n+1}-\theta_{k+1}^{n})=RHSTT\Delta t \\\\
where RHSTT= [\frac{S}{\Delta z_k}+A(\theta_{k}^{n}-\theta_{k-1}^{n})+C(\theta_{k}^{n}-\theta_{k+1}^{n})]，B=[1-(A+C)\Delta t]
\end{align}
```
Using matrix representation:
```math
\begin{bmatrix}
{B_1}&{C_1}&{0}&{0}&{0}&{\cdots}&{0}\\
{A_2}&{B_2}&{C_2}&{0}&{0}&{\cdots}&{0}\\
{0}&{A_3}&{B_3}&{C_3}&{0}&{\cdots}&{0}\\
{\vdots}&{\vdots}&{\vdots}&{\vdots}&{\vdots}&{\cdots}&{0}\\
{0}&{\cdots}&{0}&{0}&{A_{m-1}}&{B_{m-1}}&{C_{m-1}}\\
{0}&{\cdots}&{0}&{0}&{0}&{A_{m}}&{B_m}\\
\end{bmatrix}
\begin{bmatrix}
{\theta_{1}^{n+1}-\theta_{1}^{n}}\\
{\theta_{2}^{n+1}-\theta_{2}^{n}}\\
{\theta_{3}^{n+1}-\theta_{3}^{n}}\\
{\vdots}\\
{\theta_{m-1}^{n+1}-\theta_{m-1}^{n}}\\
{\theta_{m}^{n+1}-\theta_{m}^{n}}\\
\end{bmatrix}
=
\begin{bmatrix}
{RHSTT_1}\\
{RHSTT_2}\\
{RHSTT_3}\\
{\vdots}\\
{RHSTT_{m-1}}\\
{RHSTT_{m}}\\
\end{bmatrix}
```

**Simplified to:**
$$PX=D$$

The linear equation system $PX=D$ can finally be solved using the differentiable method of the machine learning platform



