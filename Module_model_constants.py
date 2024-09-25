'''
@Project ：NoahPy
@File    ：Module_model_constants.py
@Author  ：tianwb
@Date    ：2024/5/14 18:08
'''

import torch
torch.set_default_dtype(torch.float32)
EPSILON = torch.tensor(1.0e-15)
TFREEZ = torch.tensor(273.15)  # 冰点温度
LVH2O = torch.tensor(2.501e6)  # 水的汽化潜热
LSUBS = torch.tensor(2.83e6)  # 水的凝结潜热
R = torch.tensor(287.04)  # 气体常数R
RD = torch.tensor(287.04)
SIGMA = torch.tensor(5.67E-8)
CPH2O = torch.tensor(4.218E+3)
CPICE = torch.tensor(2.106E+3)
LSUBF = torch.tensor(3.335E+5) # 融化潜热
EMISSI_S = torch.tensor(0.95)
CP = torch.tensor(7 * 287 / 2)
R_D = torch.tensor(287)
XLF = torch.tensor(3.50E5)
XLV = torch.tensor(2.5E6)
RHOWATER = torch.tensor(1000.)
STBOLT = torch.tensor(5.67051E-8)
KARMAN = torch.tensor(0.4)
BARE = 19

T0 = torch.tensor(273.15)
CAIR = torch.tensor(1004.0)
CICE = torch.tensor(2.106e6)
CH2O = torch.tensor(4.2e6)

# Physical constants
G = torch.tensor(9.81)
R_D = torch.tensor(287.0)
CP = 7.0 * R_D / 2.0
R_V = torch.tensor(461.6)
CV = CP - R_D
CPV = 4.0 * R_V
CVV = CPV - R_V
CVPM = -CV / CP
CLIQ = torch.tensor(4190.0)
CICE = torch.tensor(2106.0)
PSAT = torch.tensor(610.78)
RCV = R_D / CV
RCP = R_D / CP
ROVG = R_D / G
C2 = CP * RCV
MWDRY = torch.tensor(28.966)
P1000MB = torch.tensor(100000.0)
T0 = torch.tensor(300.0)
P0 = P1000MB
CPOVCV = CP / (CP - R_D)
CVOVCP = 1.0 / CPOVCV
RVOVRD = R_V / R_D
RERADIUS = torch.tensor(1.0 / 6370.0)
ASSELIN = torch.tensor(0.025)
CB = torch.tensor(25.0)
XLV0 = torch.tensor(3.15e6)
XLV1 = torch.tensor(2370.0)
XLS0 = torch.tensor(2.905e6)
XLS1 = torch.tensor(259.532)
XLS = torch.tensor(2.85e6)
XLV = torch.tensor(2.5e6)
XLF = torch.tensor(3.5e5)
RHOWATER = torch.tensor(1000.0)
RHOSNOW = torch.tensor(100.0)
RHOAIR0 = torch.tensor(1.28)
N_CCN0 = torch.tensor(1.0e8)
DEGRAD = torch.tensor(3.1415926 / 180.0)
DPD = torch.tensor(360.0 / 365.0)
SVP1 = torch.tensor(0.6112)
SVP2 = torch.tensor(17.67)
SVP3 = torch.tensor(29.65)
SVPT0 = torch.tensor(273.15)
EP_1 = R_V / R_D - 1.0
EP_2 = R_D / R_V
KARMAN = torch.tensor(0.4)
EOMEG = torch.tensor(7.2921e-5)
STBOLT = torch.tensor(5.67051e-8)
PRANDTL = torch.tensor(1.0 / 3.0)
W_ALPHA = torch.tensor(0.3)
W_BETA = torch.tensor(1.0)
PQ0 = torch.tensor(379.90516)
EPSQ2 = torch.tensor(0.2)
A2 = torch.tensor(17.2693882)
A3 = torch.tensor(273.16)
A4 = torch.tensor(35.86)
EPSQ = torch.tensor(1.0e-12)
P608 = RVOVRD - 1.0
CLIMIT = torch.tensor(1.0e-20)
CM1 = torch.tensor(2937.4)
CM2 = torch.tensor(4.9283)
CM3 = torch.tensor(23.5518)
DEFC = torch.tensor(0.0)
DEFM = torch.tensor(99999.0)
EPSFC = torch.tensor(1.0 / 1.05)
EPSWET = torch.tensor(0.0)
FCDIF = torch.tensor(1.0 / 3.0)
FCM = torch.tensor(0.00003)
GMA = -R_D * (1.0 - RCP) * 0.5
P400 = torch.tensor(40000.0)
PHITP = torch.tensor(15000.0)
PI2 = torch.tensor(2.0 * 3.1415926)
PI1 = torch.tensor(3.1415926)
PLBTM = torch.tensor(105000.0)
PLOMD = torch.tensor(64200.0)
PMDHI = torch.tensor(35000.0)
Q2INI = torch.tensor(0.50)
RFCP = torch.tensor(0.25) / CP
RHCRIT_LAND = torch.tensor(0.75)
RHCRIT_SEA = torch.tensor(0.80)
RLAG = torch.tensor(14.8125)
RLX = torch.tensor(0.90)
SCQ2 = torch.tensor(50.0)
SLOPHT = torch.tensor(0.001)
TLC = 2.0 * 0.703972477
WA = torch.tensor(0.15)
WGHT = torch.tensor(0.35)
WPC = torch.tensor(0.075)
Z0LAND = torch.tensor(0.10)
Z0MAX = torch.tensor(0.008)
Z0SEA = torch.tensor(0.001)