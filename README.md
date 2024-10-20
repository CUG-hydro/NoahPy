# NoahPy
A new version of the backpropagation support for the land surface model.The model is based on [Noah v3.4.1](https://ral.ucar.edu/model/unified-noah-lsm) and is recoded into a differentiable model using [Pytorch](https://pytorch.org/).

## Main Processes

![NoahLSM](https://github.com/user-attachments/assets/2d062cde-37a6-49e0-ad69-8dd7b2564081)

## Basic equation

In the land surface process model, the two most core basic laws are the heat equation and the water equation, both of which are partial differential equations. Finite difference methods are often used to solve these equations, that is, they are ultimately converted into a system of equations. Therefore, the differentiable solution method of machine learning platforms (such as Pytorch and TensorFlow) can be used to enable the model to propagate gradients.

### (1) Heat equation
$$\frac{\partial}{\partial t }\rho C_p T=\frac{\partial}{\partial z}[\frac{\partial K T}{\partial z}]+Q$$

where $T_s$ represents the soil temperature; $C_s$ represents the heat capacity of the soil, $λ$ is the heat conduction of the soil, where the calculation process of the $C_s$ is as follows:
```math
C_s=\theta C_w+\left(1-\theta_s\right)C_{soil}+\left(\theta_s-\theta\right)C_{air}
```

where: $θ$ represents the soil water content; $s$ indicates the porosity of the soil; $C_{w}$, $C_{soil}$, and $C_{air}$ represent the heat capacity of water, soil substrate, and air, respectively.
Calculation formula for soil heat conduction (λ):
```math
\lambda\left(\theta\right)=K_e\left(\lambda_{sat\ }-\lambda_{dry}\right)+\lambda_{dry}
```

where $K_e$ is Kersten, $λ_{sat}$ is the heat of the saturated soil, and $λ_{dry}$ is the thermal conductivity of the dry soil.

### (2) Richards equation 
In the land surface process model, the exchange and allocation of water is crucial, which involves the balance of energy and water, and has an important impact on the soil layers. This process plays a key role in simulating characteristics such as permafrost and underground ice, as well as ecological processes such as vegetation growth and transpiration.NoahPy is described by the Richards equation for soil water movement, and its formula is as follows:
```math
\frac{\partial \theta}{\partial t} =\frac{\partial }{\partial z}[D(\theta)\frac{\partial \theta}{\partial z}] +\frac{\partial K(\theta)}{\partial z}+S
```
where: $θ$ represents the water content of the soil; $t$ stands for time; $D$ is the soil moisture diffusivity; $K$ is the conductivity of soil water, $z$ is the depth of soil; $S$ represents soil water sources and sinks (e.g., precipitation, evapotranspiration, and runoff). In this formula, the first term to the right of the equal sign represents the part of soil moisture diffusion that receives the gradient of the soil vertical water potential $Ψ$, while the second term on the right side of the equation indicates that the ground is the part of soil moisture conduction caused by gravity.
In NoahPy, the soil water conductivity $K$ and soil matrix potential $Ψ$ are calculated using the Clapp-Hornberger parameterization scheme, which is:
```math
\begin{align}
K\left(\theta\right)=K_s\left(\theta/\theta_s\right)^{2b+3} \\\\\
\Psi\left(\theta\right)=\Psi_s\left(\theta/\theta_s\right)^{-b}
\end{align}
```
where: $\Psi_s$ is the water potential of the saturated soil; $K_s$ and $\theta_s$ were saturated soil water conductivity and soil porosity, respectively. $b$ is an empirical parameter that relates to the pore size distribution of the soil matrix.

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

### RNN-wrapped main processes

![mainphysical](https://github.com/user-attachments/assets/28c3c349-2c61-4391-9713-3312ad2aeb9a)

RNN-wrapped physical processes architecture, where $\vec{S_n}$ represents the state vector at the nth moment, $\vec{X_n}$ represents the meteorological driving variable at the nth moment, $\vec{\theta_p}$ represents the model parameter vector, and $\vec{O_n}$ represents the observation vector at the nth moment.



