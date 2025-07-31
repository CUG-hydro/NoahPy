# NoahPy
A new version of the Noah land surface model (Noah LSM v3.4.1) with full backpropagation support. This model has been re-coded in  [Pytorch](https://pytorch.org/) to be fully differentiable, and it includes several key improvements to its representation of physical processes for freeze-thaw cycles. Detailed descriptions of these enhancements, along with  their applications in simulating permafrost thermal–hydrological processes, can be found in the following references:

- Zhang G, Nan Z, Hu N, Yin Z, Zhao L, Cheng G, Mu C. Qinghai-Tibet Plateau permafrost at risk in the late 21st century. Earth's Future. 2022, 10(6): e2022EF002652. https://doi.org/10.1029/2022EF002652

- Wu X, Nan Z, Zhao S, et al. Spatial modeling of permafrost distribution and properties on the Qinghai‐Tibet Plateau. Permafrost and Periglacial Processes, 2018, 29(2): 86–99. https://doi.org/10.1002/ppp.1971

- Chen H, Nan Z, Zhao L, et al. Noah modelling of the permafrost distribution and characteristics in the West Kunlun Area, Qinghai‐Tibet Plateau, China. Permafrost and Periglacial Processes, 2015, 26(2): 160–174. https://doi.org/10.1002/ppp.1841

A technical documentation, including a brief introduction to the theoretical basis, implementation details, and performance evaluation, is also included with the code for further information.

> Authors: Wenbiao Tian & Zhuotong Nan ([Orcid](https://orcid.org/0000-0002-7930-3850)) (nanzt@njnu.edu.cn)
>
> Web: [https://permalab.science](https://permalab.science)
> 
> Citation: Tian, W. & Nan, Z. (2025). wbtian/NoahPy: NoahPy v1.0.0 – Initial Release. Zenodo. [https://doi.org/10.5281/zenodo.16531046](https://doi.org/10.5281/zenodo.16531046)

July 31, 2025

## Code Structure

This folder contains the NoahPy source code. The parameter_new directory holds essential lookup tables, including those for soil parameters and other preprocessed input data.

Two versions of the NoahPy code are included:

- NoahPy.py: A procedural implementation with all functions  written independently.

- NoahPy_module.py: An object-oriented implementation where functions are encapsulated within classes. This version  generally offers better execution performance.

Both versions implement the same core processes and are interchangeable for use.

## Usage
The usage of NoahPy varies slightly depending on the version you choose:
### Using NoahPy_module (object-oriented version)
```python
from NoahPy_module import NoahPy
model = NoahPy()
Date, STC, SH2O = model.noah_main(file_name, output_flag=False, lstm_model=None)
```
- file_name: Path to your input file.
- output_flag: If set to True, the results will be written to an output file. The default is False.
- lstm_model: An optional argument for coupling with an LSTM model; Set to None if you're not using an LSTM.

### Using NoahPy.py (procedural version)
```python
from NoahPy import noah_main
Date, STC, SH2O = noah_main(file_name, output_flag=False, lstm_model=None)
```
The input parameters and output format are identical to those in the object-oriented version.

### Input Requirements
- The input forcing file must adhere to the structure used in the Noah LSM. Please refer to the provided **forcing.txt** file for an example.

- The parameter_new folder contains essential model lookup tables:

	- VEGPARM.TBL: Vegetation parameter table

	- SOILPARM.TBL: Soil parameter table

- General model parameters (originally from GENPARM.TBL) have been hardcoded into the model for simplicity.
You may modify these directly in the source code if needed.

## Main Processes

![NoahLSM](https://github.com/user-attachments/assets/2d062cde-37a6-49e0-ad69-8dd7b2564081)

### RNN-wrapped main processes

![mainphysical](https://github.com/user-attachments/assets/28c3c349-2c61-4391-9713-3312ad2aeb9a)

This diagram illustrates the RNN-wrapped physical processes architecture, where $\vec{S_n}$ represents the state vector at the nth moment, $\vec{X_n}$ represents the meteorological forcing vector at the nth moment, $\vec{\theta_p}$ represents the model parameter vector, and $\vec{O_n}$ represents the observation vector at the nth moment.

## Basic equation

In the land surface process model, the two most fundamental laws are the heat equation and soil water movement equation, both of which are partial differential equations. These equations are often solved using finite difference methods, which ultimately convert them into a system of equations. This conversion allows for the use of differentiable solution methods from machine learning platforms (such as PyTorch and TensorFlow) to enable gradient propagation within the model.

### (1) Heat equation
$$\frac{\partial}{\partial t }\rho C_p T=\frac{\partial}{\partial z}[\frac{\partial K T}{\partial z}]+Q$$

where, $T_s$ represents the soil temperature; $C_s$ represents the heat capacity of the soil, $λ$ is the heat conduction of the soil. The calculation process for $C_s$ is as follows:
```math
C_s=\theta C_w+\left(1-\theta_s\right)C_{soil}+\left(\theta_s-\theta\right)C_{air}
```

where, $θ$ represents the soil water content; $s$ indicates the porosity of the soil; $C_{w}$, $C_{soil}$, and $C_{air}$ represent the heat capacity of water, soil substrate, and air, respectively.

The fomular for calculating soil heat conduction (λ) is:
```math
\lambda\left(\theta\right)=K_e\left(\lambda_{sat\ }-\lambda_{dry}\right)+\lambda_{dry}
```

where $K_e$ is Kersten number, $λ_{sat}$ is the thermal conductivity of  saturated soil, and $λ_{dry}$ is the thermal conductivity of  dry soil.

### (2) Richards' equation 
NoahPy describes soil water movement using the Richards equation, formulated as follows:
```math
\frac{\partial \theta}{\partial t} =\frac{\partial }{\partial z}[D(\theta)\frac{\partial \theta}{\partial z}] +\frac{\partial K(\theta)}{\partial z}+S
```
where: $θ$ represents the water content of the soil; $t$ stands for time; $D$ is the soil moisture diffusivity; $K$ is the hydraulic conductivity of soil water, $z$ is the soil depth; $S$ represents soil water sources and sinks (e.g., precipitation, evapotranspiration, and runoff). In this formula, the first term to the right of the equal sign represents the part of soil moisture diffusion driven by the gradient of the soil vertical water potential $Ψ$. The second term on the right side of the equation indicates the part of soil moisture conduction caused by gravity.

In NoahPy, the soil hydaulic conductivity $K$ and soil matrix potential $Ψ$ are calculated using the Clapp-Hornberger parameterization scheme:
```math
\begin{align}
K\left(\theta\right)=K_s\left(\theta/\theta_s\right)^{2b+3} \\\\\
\Psi\left(\theta\right)=\Psi_s\left(\theta/\theta_s\right)^{-b}
\end{align}
```
where: $\Psi_s$ is the water potential of saturated soil; $K_s$ and $\theta_s$ are saturated soil water conductivity and soil porosity, respectively. $b$ is an empirical parameter that relates to the pore size distribution of soil matrix.

---
### Finite Difference Discretization

#### Soil discrete schematic diagram
![Soil discrete schematic diagram](https://github.com/user-attachments/assets/cfb240a1-f0d3-4923-8242-e5cc1b93b072)
- Pan H L, Mahrt L. Interaction between soil hydrology and boundary-layer development[J]. Boundary-Layer Meteorology, 1987, 38(1-2): 185-202. [doi: 10.1007/BF00121563](https://link.springer.com/article/10.1007/BF00121563).
- Kalnay E, Kanamitsu M. Time Schemes for Strongly Nonlinear Damping Equations[J]. Monthly Weather Review, 1988, 116(10): 1945-1958. [doi: 10.1175/1520-0493(1988)116<1945:TSFSND>2.0.CO;2](https://journals.ametsoc.org/view/journals/mwre/116/10/1520-0493_1988_116_1945_tsfsnd_2_0_co_2.xml).

The finite difference form of the equation can be derived using the above discretization scheme and the time scheme "D" (IMPLICIT STATE, EXPLICIT COEFFICIENT)  as presented in Section 2 of Kalnay & Kanamitsu:

**We obtain:**

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

**This simplified to:**
$$PX=D$$

This linear equation system, $PX=D$, can then be solved using the differentiable method available in the machine learning platforms.





