'''
@Project ：NoahPy_New 
@File    ：Module.py
@Author  ：tianwb
@Date    ：2024/6/7 下午3:28 
'''
import cProfile
import io
import os
import pstats
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, tensor, Tensor
# from .plot_simulate import plot_timeseries


class ConstantsModule(nn.Module):
    def __init__(self):
        super(ConstantsModule, self).__init__()
        self.EPSILON = torch.tensor(1.0e-15)
        self.TFREEZ = torch.tensor(273.15) 
        self.LVH2O = torch.tensor(2.501e6)  
        self.LSUBS = torch.tensor(2.83e6) 
        self.R = torch.tensor(287.04)  
        self.RD = torch.tensor(287.04)
        self.SIGMA = torch.tensor(5.67E-8)
        self.CPH2O = torch.tensor(4.218E+3)
        self.CPICE = torch.tensor(2.106E+3)
        self.LSUBF = torch.tensor(3.335E+5) 
        self.EMISSI_S = torch.tensor(0.95)
        self.CP = torch.tensor(7 * 287 / 2)
        self.R_D = torch.tensor(287)
        self.XLF = torch.tensor(3.50E5)
        self.XLV = torch.tensor(2.5E6)
        self.RHOWATER = torch.tensor(1000.)
        self.STBOLT = torch.tensor(5.67051E-8)
        self.KARMAN = torch.tensor(0.4)
        self.BARE = 19

        self.T0 = torch.tensor(273.15)
        self.CAIR = torch.tensor(1004.0)
        self.CICE = torch.tensor(2.106e6)
        self.CH2O = torch.tensor(4.2e6)

        self.G = torch.tensor(9.81)
        self.R_D = torch.tensor(287.0)
        self.CP = 7.0 * self.R_D / 2.0
        self.R_V = torch.tensor(461.6)
        self.CV = self.CP - self.R_D
        self.CPV = 4.0 * self.R_V
        self.CVV = self.CPV - self.R_V
        self.CVPM = -self.CV / self.CP
        self.CLIQ = torch.tensor(4190.0)
        self.CICE = torch.tensor(2106.0)
        self.PSAT = torch.tensor(610.78)
        self.RCV = self.R_D / self.CV
        self.RCP = self.R_D / self.CP
        self.ROVG = self.R_D / self.G
        self.C2 = self.CP * self.RCV
        self.MWDRY = torch.tensor(28.966)
        self.P1000MB = torch.tensor(100000.0)
        self.T0 = torch.tensor(300.0)
        self.P0 = self.P1000MB
        self.CPOVCV = self.CP / (self.CP - self.R_D)
        self.CVOVCP = 1.0 / self.CPOVCV
        self.RVOVRD = self.R_V / self.R_D
        self.RERADIUS = torch.tensor(1.0 / 6370.0)
        self.ASSELIN = torch.tensor(0.025)
        self.CB = torch.tensor(25.0)
        self.XLV0 = torch.tensor(3.15e6)
        self.XLV1 = torch.tensor(2370.0)
        self.XLS0 = torch.tensor(2.905e6)
        self.XLS1 = torch.tensor(259.532)
        self.XLS = torch.tensor(2.85e6)
        self.XLV = torch.tensor(2.5e6)
        self.XLF = torch.tensor(3.5e5)
        self.RHOWATER = torch.tensor(1000.0)
        self.RHOSNOW = torch.tensor(100.0)
        self.RHOAIR0 = torch.tensor(1.28)
        self.N_CCN0 = torch.tensor(1.0e8)
        self.DEGRAD = torch.tensor(3.1415926 / 180.0)
        self.DPD = torch.tensor(360.0 / 365.0)
        self.SVP1 = torch.tensor(0.6112)
        self.SVP2 = torch.tensor(17.67)
        self.SVP3 = torch.tensor(29.65)
        self.SVPT0 = torch.tensor(273.15)
        self.EP_1 = self.R_V / self.R_D - 1.0
        self.EP_2 = self.R_D / self.R_V
        self.KARMAN = torch.tensor(0.4)
        self.EOMEG = torch.tensor(7.2921e-5)
        self.STBOLT = torch.tensor(5.67051e-8)
        self.PRANDTL = torch.tensor(1.0 / 3.0)
        self.W_ALPHA = torch.tensor(0.3)
        self.W_BETA = torch.tensor(1.0)
        self.PQ0 = torch.tensor(379.90516)
        self.EPSQ2 = torch.tensor(0.2)
        self.A2 = torch.tensor(17.2693882)
        self.A3 = torch.tensor(273.16)
        self.A4 = torch.tensor(35.86)
        self.EPSQ = torch.tensor(1.0e-12)
        self.P608 = self.RVOVRD - 1.0
        self.CLIMIT = torch.tensor(1.0e-20)
        self.CM1 = torch.tensor(2937.4)
        self.CM2 = torch.tensor(4.9283)
        self.CM3 = torch.tensor(23.5518)
        self.DEFC = torch.tensor(0.0)
        self.DEFM = torch.tensor(99999.0)
        self.EPSFC = torch.tensor(1.0 / 1.05)
        self.EPSWET = torch.tensor(0.0)
        self.FCDIF = torch.tensor(1.0 / 3.0)
        self.FCM = torch.tensor(0.00003)
        self.GMA = -self.R_D * (1.0 - self.RCP) * 0.5
        self.P400 = torch.tensor(40000.0)
        self.PHITP = torch.tensor(15000.0)
        self.PI2 = torch.tensor(2.0 * 3.1415926)
        self.PI1 = torch.tensor(3.1415926)
        self.PLBTM = torch.tensor(105000.0)
        self.PLOMD = torch.tensor(64200.0)
        self.PMDHI = torch.tensor(35000.0)
        self.Q2INI = torch.tensor(0.50)
        self.RFCP = torch.tensor(0.25) / self.CP
        self.RHCRIT_LAND = torch.tensor(0.75)
        self.RHCRIT_SEA = torch.tensor(0.80)
        self.RLAG = torch.tensor(14.8125)
        self.RLX = torch.tensor(0.90)
        self.SCQ2 = torch.tensor(50.0)
        self.SLOPHT = torch.tensor(0.001)
        self.TLC = torch.tensor(2.0 * 0.703972477)
        self.WA = torch.tensor(0.15)
        self.WGHT = torch.tensor(0.35)
        self.WPC = torch.tensor(0.075)
        self.Z0LAND = torch.tensor(0.10)
        self.Z0MAX = torch.tensor(0.008)
        self.Z0SEA = torch.tensor(0.001)


class SFCDIFModule(ConstantsModule):
    def __init__(self):
        super(SFCDIFModule, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.ITRMX = torch.tensor(5)

        self.EXCML = torch.tensor(0.0001)
        self.EXCMS = torch.tensor(0.0001)
        self.VKARMAN = torch.tensor(0.4)
        self.ZTFC = torch.tensor(1.0)
        self.ELOCP = torch.tensor(2.72e6) / self.CP
        self.EPSU2 = torch.tensor(1.0e-6)
        self.EPSUST = torch.tensor(1.0e-9)
        self.SQVISC = torch.tensor(258.2)
        self.RIC = torch.tensor(0.505)
        self.EPSZT = torch.tensor(1.0e-28)
        self.RD = tensor(287.0)
        self.KZTM = 10001
        self.KZTM2 = self.KZTM - 2

        self.WWST = torch.tensor(1.2)
        self.WWST2 = self.WWST * self.WWST

        self.PSIM2 = torch.zeros(self.KZTM)
        self.PSIH2 = torch.zeros(self.KZTM)
        self.ZTMAX2 = torch.zeros(0)
        self.DZETA2 = torch.zeros(0)
        self.ZTMIN2 = torch.tensor(-5.0)

    @torch.jit.export
    def MYJSFCINIT(self):
        # global ZTMIN2, ZTMAX2, DZETA2, PSIH2, PSIM2
        # Parameter definitions
        PIHF = torch.pi / 2.
        EPS = tensor(1.0e-6)
        ZTMIN1 = tensor(-5.0)

        # Variable initialization
        ZTMAX1 = tensor(1.0)
        self.ZTMAX2 = tensor(1.0)
        ZRNG1 = ZTMAX1 - ZTMIN1
        ZRNG2 = self.ZTMAX2 - self.ZTMIN2
        DZETA1 = ZRNG1 / (self.KZTM - 1)
        self.DZETA2 = ZRNG2 / (self.KZTM - 1)
        ZETA1 = ZTMIN1.clone()
        ZETA2 = self.ZTMIN2.clone()

        # Function definition loop
        for K in range(1, self.KZTM + 1):
            if ZETA2 < 0.:
                X = torch.sqrt(torch.sqrt(1. - 16. * ZETA2))
                self.PSIM2[K - 1] = -2. * torch.log((X + 1.) / 2.) - torch.log((X * X + 1.) / 2.) + 2. * torch.arctan(
                    X) - PIHF
                self.PSIH2[K - 1] = -2. * torch.log((X * X + 1.) / 2.)
            else:
                self.PSIM2[K - 1] = 0.7 * ZETA2 + 0.75 * ZETA2 * (6. - 0.35 * ZETA2) * torch.exp(-0.35 * ZETA2)
                self.PSIH2[K - 1] = 0.7 * ZETA2 + 0.75 * ZETA2 * (6. - 0.35 * ZETA2) * torch.exp(-0.35 * ZETA2)

            if K == self.KZTM:
                ZTMAX1 = ZETA1
                self.ZTMAX2 = ZETA2

            ZETA1 = ZETA1 + DZETA1
            ZETA2 = ZETA2 + self.DZETA2

        ZTMAX1 -= EPS
        self.ZTMAX2 -= EPS

    @torch.jit.export
    def SFCDIF_MYJ(self, ZSL, ZSL_WIND, Z0, Z0BASE, SFCPRS, TZ0, TLOW, QZ0, QLOW, SFCSPD, CZIL, AKMS, AKHS, IZ0TLND):
        THLOW = TLOW * (self.P0 / SFCPRS) ** self.RCP
        THZ0 = TZ0 * (self.P0 / SFCPRS) ** self.RCP
        THELOW = THLOW
        CXCHL = self.EXCML / ZSL
        BTGX = self.G / THLOW
        ELFC = self.VKARMAN * BTGX
        BTGH = BTGX * 1000.
        THM = (THELOW + THZ0) * 0.5
        TEM = (TLOW + TZ0) * 0.5
        A = THM * self.P608
        B = (self.ELOCP / TEM - 1. - self.P608) * THM
        CWMLOW = torch.tensor(0.0)
        DTHV = ((THELOW - THZ0) * ((QLOW + QZ0 + CWMLOW) * (0.5 * self.P608) + 1.) + (
                QLOW - QZ0 + CWMLOW) * A + CWMLOW * B)
        DU2 = torch.max(SFCSPD * SFCSPD, self.EPSU2)
        RIB = BTGX * DTHV * ZSL_WIND * ZSL_WIND / DU2 / ZSL
        ZU = Z0
        ZT = ZU * self.ZTFC
        ZSLU = ZSL_WIND + ZU
        RZSU = ZSLU / ZU
        RLOGU = torch.log(RZSU)
        ZSLT = ZSL + ZU
        CZIL_LOCAL = 10.0 ** (-0.40 * (Z0 / 0.07))
        ZILFC = torch.where(torch.eq(IZ0TLND, 0), -CZIL * self.VKARMAN * self.SQVISC,
                            -CZIL_LOCAL * self.VKARMAN * self.SQVISC)
        CZETMAX = 10.
        ZZIL = torch.where(torch.gt(DTHV, 0),
                           torch.where(torch.lt(RIB, self.RIC),
                                       ZILFC * (1.0 + (RIB / self.RIC) * (RIB / self.RIC) * CZETMAX),
                                       ZILFC * (1.0 + CZETMAX)), ZILFC)
        WSTAR2 = torch.where(BTGH * AKHS * DTHV != 0.0, self.WWST2 * torch.abs(BTGH * AKHS * DTHV) ** (2.0 / 3.0),
                             tensor(0.0))

        USTAR = torch.max(torch.sqrt(AKMS * torch.sqrt(DU2 + WSTAR2)), self.EPSUST)
        ITRMX = 5
        for ITR in range(0, ITRMX):
            ZT = torch.max(torch.exp(ZZIL * torch.sqrt(USTAR * Z0BASE)) * Z0BASE, self.EPSZT)
            RZST = ZSLT / ZT
            RLOGT = torch.log(RZST)
            RLMO = ELFC * AKHS * DTHV / USTAR ** 3
            ZETALU = ZSLU * RLMO
            ZETALT = ZSLT * RLMO
            ZETAU = ZU * RLMO
            ZETAT = ZT * RLMO

            ZETALU = torch.min(torch.max(ZETALU, self.ZTMIN2), self.ZTMAX2)
            ZETALT = torch.min(torch.max(ZETALT, self.ZTMIN2), self.ZTMAX2)
            ZETAU = torch.min(torch.max(ZETAU, self.ZTMIN2 / RZSU), self.ZTMAX2 / RZSU)
            ZETAT = torch.min(torch.max(ZETAT, self.ZTMIN2 / RZST), self.ZTMAX2 / RZST)
            #
            RZ = (ZETAU - self.ZTMIN2) / self.DZETA2
            K = torch.floor(RZ).int()
            RDZT = RZ - K.float()
            K = torch.clamp(K, tensor(0), tensor(self.KZTM2))
            PSMZ = (self.PSIM2[K + 1] - self.PSIM2[K]) * RDZT + self.PSIM2[K]
            #
            RZ = (ZETALU - self.ZTMIN2) / self.DZETA2
            K = torch.floor(RZ).int()
            RDZT = RZ - K.float()
            K = torch.clamp(K, tensor(0), tensor(self.KZTM2))
            PSMZL = (self.PSIM2[K + 1] - self.PSIM2[K]) * RDZT + self.PSIM2[K]
            #
            SIMM = PSMZL - PSMZ + RLOGU
            #
            RZ = (ZETAT - self.ZTMIN2) / self.DZETA2
            K = torch.floor(RZ).int()
            RDZT = RZ - K.float()
            K = torch.clamp(K, tensor(0), tensor(self.KZTM2))
            PSHZ = (self.PSIH2[K + 1] - self.PSIH2[K]) * RDZT + self.PSIH2[K]
            #
            RZ = (ZETALT - self.ZTMIN2) / self.DZETA2
            K = torch.floor(RZ).int()
            RDZT = RZ - K.float()
            K = torch.clamp(K, tensor(0), tensor(self.KZTM2))
            PSHZL = (self.PSIH2[K + 1] - self.PSIH2[K]) * RDZT + self.PSIH2[K]
            #
            SIMH = PSHZL - PSHZ + RLOGT
            USTARK = USTAR * self.VKARMAN
            AKMS = torch.max(USTARK / SIMM, CXCHL)
            AKHS = torch.max(USTARK / SIMH, CXCHL)

            WSTAR2 = torch.where(
                DTHV <= 0.0,
                self.WWST2 * torch.abs(BTGH * AKHS * DTHV) ** (2.0 / 3.0),
                torch.tensor(0.0)
            )
            USTAR = torch.max(torch.sqrt(AKMS * torch.sqrt(DU2 + WSTAR2)), self.EPSUST)
        return RIB, AKMS, AKHS

    @torch.jit.export
    def SFCDIF_MYJ_Y08(self, z0m, zm, zh, wspd1, tsfc, tair, qair, psfc):
        # Parameters
        excm = tensor(0.001)
        aa = tensor(0.007)
        p0 = tensor(1.0e5)

        # Local variables
        wspd = torch.max(wspd1, tensor(0.01))
        rhoair = psfc / (self.RD * tair * (1 + 0.61 * qair))
        ptair = tair * (psfc / (psfc - rhoair * 9.81 * zh)) ** self.RCP
        ptsfc = tsfc

        pt1 = ptair

        c_u = 0.4 / torch.log(zm / z0m)  # von Karman constant / log(zm / z0m)
        c_pt = 0.4 / torch.log(zh / z0m)  # von Karman constant / log(zh / z0m)

        tstr = c_pt * (pt1 - ptsfc)  # tstr for stable case
        ustr = c_u * wspd  # ustr for stable case

        lmo = ptair * ustr ** 2 / (0.4 * 9.81 * tstr)  # Monin Obukhov length

        nu = 1.328e-5 * (p0 / psfc) * (pt1 / 273.15) ** 1.754  # viscosity of air
        ribbb = tensor(0)
        for i in range(3):
            z0h = self.z0mz0h(zh, z0m, nu, ustr, tstr)
            c_u, c_pt, ribbb = self.flxpar(zm, zh, z0m, z0h, wspd, ptsfc, pt1)

            ustr = c_u * wspd
            tstr = c_pt * (pt1 - ptsfc)

        c_pt = torch.where((torch.abs(ptair - ptsfc) < 0.001), c_pt, tstr / (ptair - ptsfc))

        ra = 1 / (ustr * c_pt)

        chh = 1.0 / ra

        chh = torch.max(chh, excm * (1.0 / zm))
        # chh = max(chh, aa)  # Uncomment if needed

        return ribbb, chh

    def z0mz0h(self, zh, z0m, nu, ustr, tstr):
        # Constants
        a = tensor(70.0)

        # Parameters
        b = tensor(-7.2)

        # Calculate z0h
        z0h = a * nu / ustr * torch.exp(b * torch.sqrt(ustr) * torch.sqrt(torch.sqrt(torch.abs(-tstr))))
        z0h = torch.min(zh / 10, torch.max(z0h, tensor(1.0E-10)))

        return z0h

    def flxpar(self, zm, zh, z0m, z0h, wspd, ptsfc, pt1):
        lmo, ribb1 = self.MOlength(zm, zh, z0m, z0h, wspd, ptsfc, pt1)

        c_u, c_pt = self.CuCpt(lmo, z0m, z0h, zm, zh)

        return c_u, c_pt, ribb1

    def MOlength(self, zm, zh, z0m, z0h, wspd, ptsfc, pt1):
        # Constants
        g = tensor(9.81)
        prantl01 = tensor(1.0)
        prantl02 = tensor(0.95)
        betah = tensor(8.0)
        betam = tensor(5.3)
        gammah = tensor(11.6)
        gammam = tensor(19.0)

        bulkri = (g / pt1) * (pt1 - ptsfc) * (zm - z0m) / (wspd ** 2)

        if bulkri < 0.0:
            bulkri = torch.max(bulkri, tensor(-10.0))

            d = bulkri / prantl02
            numerator = d * ((torch.log(zm / z0m)) ** 2 / torch.log(zh / z0h)) * (1 / (zm - z0m))

            a = torch.log(-d)
            b = torch.log(torch.log(zm / z0m))
            c = torch.log(torch.log(zh / z0h))

            p = 0.03728 - 0.093143 * a - 0.24069 * b + 0.30616 * c + \
                0.017131 * a ** 2 + 0.037666 * a * b - 0.084598 * b ** 2 - 0.016498 * a * c + \
                0.1828 * b * c - 0.12587 * c ** 2
            p = torch.max(tensor(0.0), p)
            coef = d * gammam ** 2 / 8 / gammah * (zm - z0m) / (zh - z0h)
            if torch.abs(1 - coef * p) < 1.0e-6:
                raise ValueError('Stop in similarity')
            lmo = numerator / (1 - coef * p)

        else:
            bulkri = torch.min(torch.min(bulkri, tensor(0.2)),
                               prantl01 * betah * (1 - z0h / zh) / betam ** 2 / (1 - z0m / zm) - 0.05)
            d = bulkri / prantl01
            a = d * betam ** 2 * (zm - z0m) - betah * (zh - z0h)
            b = 2 * d * betam * torch.log(zm / z0m) - torch.log(zh / z0h)
            c = d * torch.log(zm / z0m) ** 2 / (zm - z0m)
            lmo = (-b - torch.sqrt(torch.abs(b ** 2 - 4 * a * c))) / (2 * a)

        if 0 < lmo < 1.0e-6:
            lmo = tensor(1.0e-6)
        elif 0 > lmo > -1.0e-6:
            lmo = tensor(-1.0e-6)

        lmo = 1 / lmo

        return lmo, bulkri

    def CuCpt(self, lmo, zm1, zh1, zm2, zh2):
        # Constants
        kv = 0.4
        gammam = 19.0
        gammah = 11.6
        prantl01 = 1.0
        prantl02 = 0.95
        betam = 5.3
        betah = 8.0

        if lmo < 0:
            xx2 = torch.sqrt(torch.sqrt(1 - gammam * zm2 / lmo))
            xx1 = torch.sqrt(torch.sqrt(1 - gammam * zm1 / lmo))
            psim = 2 * torch.log((1 + xx2) / (1 + xx1)) + torch.log((1 + xx2 ** 2) / (1 + xx1 ** 2)) \
                   - 2 * torch.atan(xx2) + 2 * torch.atan(xx1)

            yy2 = torch.sqrt(1 - gammah * zh2 / lmo)
            yy1 = torch.sqrt(1 - gammah * zh1 / lmo)
            psih = 2 * torch.log((1 + yy2) / (1 + yy1))

            uprf = torch.max(torch.log(zm2 / zm1) - psim, 0.50 * torch.log(zm2 / zm1))
            ptprf = torch.max(torch.log(zh2 / zh1) - psih, 0.33 * torch.log(zm2 / zm1))

            c_u = kv / uprf
            c_pt = kv / (ptprf * prantl02)
        else:
            psim = -betam * (zm2 - zm1) / lmo
            psih = -betah * (zh2 - zh1) / lmo
            psim = max(-betam, psim)
            psih = max(-betah, psih)
            uprf = torch.min(torch.log(zm2 / zm1) - psim, 2.0 * torch.log(zm2 / zm1))
            ptprf = torch.min(torch.log(zh2 / zh1) - psih, 2.0 * torch.log(zm2 / zm1))
            c_u = kv / uprf
            c_pt = kv / (ptprf * prantl01)

        return c_u, c_pt


class NoahLSMModule(nn.Module):
    def __init__(self, file_name:str, VEGTYP, ZSOIL, SLDPTH, SHDMIN, SHDMAX, STYPE, DT, TBOT, SLOPETYPE,
                 RDLAI2D=False, USEMONALB=False, grad_soil_param=None):
        super().__init__()
        self.k_time = 0
        self.OPT_FRZ = 1
        self.file_name = file_name
        # options for frozen soil permeability
        # 1 -> Noah default, nonlinear effects, less permeable (old)  //no ice
        # 2 -> New parametric scheme,linear effects, more permeable (Niu and Yang, 2006, JHM)  //have ice
        self.OPT_INF = 2  # (suggested 2 ice)

        # options for soil thermal conductivity
        # 1 -> Noah default,   // no gravel
        # 2 -> New parametric scheme,   //have gravel & bedrock
        self.OPT_TCND = 2  # (suggested 2 gravel)

        self.EMISSI_S = torch.tensor(0.95)
        self.CP = torch.tensor(7 * 287 / 2)
        self.BARE = 19
        self.TFREEZ = torch.tensor(273.15) 

        gen_parameters = {
            'SBETA_DATA': tensor(-2.0),
            'FXEXP_DATA': tensor(2.0),
            'CSOIL_DATA': tensor(2.00E+6),
            'SALP_DATA': tensor(2.6),
            'REFDK_DATA': tensor(2.0E-6),
            'REFKDT_DATA': tensor(3.0),
            'FRZK_DATA': tensor(0.15),
            'ZBOT_DATA': tensor(-40.0),
            'CZIL_DATA': tensor(0.1),
            'SMLOW_DATA': tensor(0.5),
            'SMHIGH_DATA': tensor(3.0),
            'LVCOEF_DATA': tensor(0.5),
            'TOPT_DATA': tensor(298.0),
            'CMCMAX_DATA': tensor(0.5E-3),
            'CFACTR_DATA': tensor(0.5),
            'RSMAX_DATA': tensor(5000.0),
            'BARE': 19,
            'NATURAL': tensor(5),
            'SLOPE_DATA': tensor([0.1, 0.6, 1.0, 0.35, 0.55, 0.8, 0.63, 0.0, 0.0])
        }
        # SFCDIFModule_path = os.path.join(current_dir, "SFCDIFModule.pt")
        # NoahLSMModule_path = os.path.join(current_dir, "NoahLSMModule.pt")
        # self.SFCDIFModule = torch.jit.load(SFCDIFModule_path)
        # self.NoahLSMModule = torch.jit.load(NoahLSMModule_path)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        veg_param_path = os.path.join(current_dir, "parameter_new/VEGPARM.TBL")

        veg_parameter = pd.read_csv(veg_param_path, sep=r',\s*', engine='python', header=0, index_col=0,
                                    usecols=range(16),
                                    dtype=np.float32)
        target_columns = ['NROOT', 'RS', 'RGL', 'HS', 'SNUP', 'LAIMIN', 'LAIMAX', 'EMISSMIN', 'EMISSMAX', 'ALBEDOMIN',
                          'ALBEDOMAX', 'Z0MIN', 'Z0MAX']
        self.veg_parameter = tensor(veg_parameter[target_columns].to_numpy(), dtype=torch.float32)
        # self.veg_parameter = veg_parameter.to_dict(orient='index')
        self.ZSOIL = ZSOIL
        self.SLDPTH = SLDPTH
        self.STYPE = STYPE
        self.DT = tensor(DT)
        self.TBOT = TBOT
        NSOIL = ZSOIL.size(0)
        self.DENOM2 = torch.zeros_like(ZSOIL)
        self.DENOM2[0] = -ZSOIL[0]
        self.DENOM2[1:] = ZSOIL[:-1] - ZSOIL[1:]

        self.DDZ2 = torch.zeros(NSOIL - 1)
        self.DDZ2[0] = (0 - self.ZSOIL[1]) * 0.5
        self.DDZ2[1:] = (self.ZSOIL[:-2] - self.ZSOIL[2:]) * 0.5

        self.SHDMIN = SHDMIN
        self.SHDMAX = SHDMAX
        self.RDLAI2D = tensor(RDLAI2D)
        self.USEMONALB = tensor(USEMONALB)
        self.VEGTYP = VEGTYP
        # NUMBER,SHDFAC,NROOT,RS,RGL,HS,SNUP,MAXALB,LAIMIN,LAIMAX,EMISSMIN,EMISSMAX,ALBEDOMIN,ALBEDOMAX,Z0MIN,Z0MAX,TYPE
        self.NROOT = max(int(self.veg_parameter[VEGTYP - 1, 0].item()), 1)
        self.RSMIN = self.veg_parameter[VEGTYP - 1, 1]
        self.RGL = self.veg_parameter[VEGTYP - 1, 2]
        self.HS = self.veg_parameter[VEGTYP - 1, 3]
        self.SNUP = self.veg_parameter[VEGTYP - 1, 4]

        self.LAIMIN = self.veg_parameter[VEGTYP - 1, 5]
        self.LAIMAX = self.veg_parameter[VEGTYP - 1, 6]

        self.EMISSMIN = self.veg_parameter[VEGTYP - 1, 7]
        self.EMISSMAX = self.veg_parameter[VEGTYP - 1, 8]

        self.ALBEDOMIN = self.veg_parameter[VEGTYP - 1, 9]
        self.ALBEDOMAX = self.veg_parameter[VEGTYP - 1, 10]

        self.Z0MIN = self.veg_parameter[VEGTYP - 1, 11]
        self.Z0MAX = self.veg_parameter[VEGTYP - 1, 12]

        self.RTDIS = - self.SLDPTH[:self.NROOT] / self.ZSOIL[self.NROOT - 1]

        self.CSOIL = gen_parameters['CSOIL_DATA']
        self.ZBOT = gen_parameters['ZBOT_DATA']
        self.SALP = gen_parameters['SALP_DATA']
        self.SBETA = gen_parameters['SBETA_DATA']
        self.REFDK = gen_parameters['REFDK_DATA']
        self.FRZK = gen_parameters['FRZK_DATA']
        self.FXEXP = gen_parameters['FXEXP_DATA']
        self.REFKDT = gen_parameters['REFKDT_DATA']
        self.PTU = tensor(0.0)
        # KDT, FRAFACT, FRZX, NROOT
        self.CZIL = gen_parameters['CZIL_DATA']
        self.SLOPE_DATA = tensor([0.1, 0.6, 1.0, 0.35, 0.55, 0.8, 0.63, 0.0, 0.0])
        self.SLOPE = self.SLOPE_DATA[SLOPETYPE - 1]
        self.LVCOEF = gen_parameters['LVCOEF_DATA']
        self.TOPT = gen_parameters['TOPT_DATA']
        self.CMCMAX = gen_parameters['CMCMAX_DATA']
        self.CFACTR = gen_parameters['CFACTR_DATA']
        self.RSMAX = gen_parameters['RSMAX_DATA']
        self.Soil_Parameter = self.SoilParam()
        if grad_soil_param is not None:
            self.use_grad_soil_param = True
            self.grad_soil_param = grad_soil_param
        else:
            self.use_grad_soil_param = False
            self.grad_soil_param = (tensor(0), tensor(0), tensor(0), tensor(0))

    @torch.jit.export
    def set_parameter(self, VEGTYP: int, ZSOIL, SLDPTH, SHDMIN, SHDMAX, STYPE, DT: int, TBOT, SLOPETYPE: int,
                      RDLAI2D:bool=False, USEMONALB:bool=False):
        self.VEGTYP = VEGTYP
        self.ZSOIL = ZSOIL
        self.SLDPTH = SLDPTH
        self.STYPE = STYPE
        self.DT = tensor(DT)
        self.TBOT = TBOT
        NSOIL = ZSOIL.size(0)
        self.DENOM2 = torch.zeros_like(ZSOIL)
        self.DENOM2[0] = -ZSOIL[0]
        self.DENOM2[1:] = ZSOIL[:-1] - ZSOIL[1:]
        self.DDZ2 = torch.zeros(NSOIL - 1)
        self.DDZ2[0] = (0 - self.ZSOIL[1]) * 0.5
        self.DDZ2[1:] = (self.ZSOIL[:-2] - self.ZSOIL[2:]) * 0.5

        self.SHDMIN = SHDMIN
        self.SHDMAX = SHDMAX
        self.RDLAI2D = tensor(RDLAI2D)
        self.USEMONALB = tensor(USEMONALB)
        self.VEGTYP = VEGTYP
        self.NROOT = max(int(self.veg_parameter[VEGTYP - 1, 0].item()), 1)
        self.RSMIN = self.veg_parameter[VEGTYP - 1, 1]
        self.RGL = self.veg_parameter[VEGTYP - 1, 2]
        self.HS = self.veg_parameter[VEGTYP - 1, 3]
        self.SNUP = self.veg_parameter[VEGTYP - 1, 4]

        self.LAIMIN = self.veg_parameter[VEGTYP - 1, 5]
        self.LAIMAX = self.veg_parameter[VEGTYP - 1, 6]

        self.EMISSMIN = self.veg_parameter[VEGTYP - 1, 7]
        self.EMISSMAX = self.veg_parameter[VEGTYP - 1, 8]

        self.ALBEDOMIN = self.veg_parameter[VEGTYP - 1, 9]
        self.ALBEDOMAX = self.veg_parameter[VEGTYP - 1, 10]

        self.Z0MIN = self.veg_parameter[VEGTYP - 1, 11]
        self.Z0MAX = self.veg_parameter[VEGTYP - 1, 12]
        self.SLOPE = self.SLOPE_DATA[SLOPETYPE - 1]

    @torch.jit.export
    def set_grad_soil_param(self, grad_soil_param: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        self.grad_soil_param = grad_soil_param
        self.use_grad_soil_param = True

    @torch.jit.export
    def reset_grad_soil_param(self):
        """重置 grad_soil_param"""
        self.grad_soil_param = (torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0))
        self.use_grad_soil_param = False

    class SoilParam(nn.Module):
        def __init__(self):
            super(NoahLSMModule.SoilParam, self).__init__()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            soil_param_path = os.path.join(current_dir, "parameter_new\SOILPARM.TBL")
            soil_parameter = pd.read_csv(soil_param_path, sep=r',\s*', engine='python', header=0, index_col=0,
                                         usecols=range(11),
                                         dtype=np.float32)
            target_columns = ['BB', 'MAXSMC', 'SATDK', 'SATPSI', 'QTZ']
            soil_parameters = soil_parameter[target_columns]
            self.BB = nn.Parameter(tensor(soil_parameters['BB'].to_numpy(), dtype=torch.float32))
            self.MAXSMC = nn.Parameter(tensor(soil_parameters['MAXSMC'].to_numpy(), dtype=torch.float32))
            self.SATDK = nn.Parameter(tensor(soil_parameters['SATDK'].to_numpy(), dtype=torch.float32))
            self.SATPSI = nn.Parameter(tensor(soil_parameters['SATPSI'].to_numpy(), dtype=torch.float32))
            self.QTZ = nn.Parameter(tensor(soil_parameters['QTZ'].to_numpy(), dtype=torch.float32))

        def get_by_index(self, indices, required_grad: bool = True):
            index = indices - 1
            BB = self.BB[index].squeeze()
            MAXSMC = self.MAXSMC[index].squeeze()
            SATDK = self.SATDK[index].squeeze()
            SATPSI = self.SATPSI[index].squeeze()
            if required_grad:
                return BB, MAXSMC, SATDK, SATPSI
            else:
                return BB.detach(), MAXSMC.detach(), SATDK.detach(), SATPSI.detach()

        def get_MAXSMC_by_index(self, indices, required_grad: bool = True):
            index = indices - 1
            MAXSMC = self.MAXSMC[index].squeeze()
            if required_grad:
                return MAXSMC
            else:
                return MAXSMC.detach()

        def get_SATDK_by_index(self, indices, required_grad: bool = True):
            index = indices - 1
            SATDK = self.SATDK[index].squeeze()
            if required_grad:
                return SATDK
            else:
                return SATDK.detach()

        def get_SATPSI_by_index(self, indices, required_grad: bool = True):
            index = indices - 1
            SATPSI = self.SATPSI[index].squeeze()
            if required_grad:
                return SATPSI
            else:
                return SATPSI.detach()

        def get_QTZ_by_index(self, indices, required_grad: bool = True):
            index = indices - 1
            QTZ = self.QTZ[index].squeeze()
            if required_grad:
                return QTZ
            else:
                return QTZ.detach()

        def get_all_by_index(self, indices, required_grad: bool = True):
            index = indices - 1
            bb = self.BB[index].squeeze()
            maxSMC = self.MAXSMC[index].squeeze()
            SATDK = self.SATDK[index].squeeze()
            SATPSI = self.SATPSI[index].squeeze()
            QTZ = self.QTZ[index].squeeze()
            if required_grad:
                return bb, maxSMC, SATDK, SATPSI, QTZ
            else:
                return bb.detach(), maxSMC.detach(), SATDK.detach(), SATPSI.detach(), QTZ.detach()

    def read_surface_param(self):
        # DKSAT, SMCMAX, SMCREF , SMCDRY
        if self.use_grad_soil_param:
            BEXP, SMCMAX, DKSAT, PSISAT = [item[0] for item in self.grad_soil_param]
        else:
            BEXP, SMCMAX, DKSAT, PSISAT = self.Soil_Parameter.get_by_index(self.STYPE[0])
        QUARTZ = self.Soil_Parameter.get_QTZ_by_index(self.STYPE[0])
        REFSMC1 = SMCMAX * (5.79E-9 / DKSAT) ** (1 / (2 * BEXP + 3))
        SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
        WLTSMC1 = SMCMAX * (200. / PSISAT) ** (-1. / BEXP)
        SMCWLT = 0.5 * WLTSMC1
        SMCDRY = SMCWLT
        return DKSAT, SMCMAX, SMCREF, SMCDRY, QUARTZ

    def read_root_param(self):
        if self.use_grad_soil_param:
            BEXP, SMCMAX, DKSAT, PSISAT = [item[:self.NROOT] for item in self.grad_soil_param]
        else:
            BEXP, SMCMAX, DKSAT, PSISAT = self.Soil_Parameter.get_by_index(self.STYPE[:self.NROOT])
        REFSMC1 = SMCMAX * (5.79E-9 / DKSAT) ** (1 / (2 * BEXP + 3))
        SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
        WLTSMC1 = SMCMAX * (200. / PSISAT) ** (-1. / BEXP)
        SMCWLT = 0.5 * WLTSMC1
        return SMCWLT, SMCREF

    def read_all_param(self):
        train_layer_size = 10
        if self.use_grad_soil_param:
            first_ten_layer_BEXP, first_ten_layer_SMCMAX, first_ten_layer_DKSAT, first_ten_layer_PSISAT = self.grad_soil_param
            other_layer_BEXP, other_layer_SMCMAX, other_layer_DKSAT, other_layer_PSISAT = self.Soil_Parameter.get_by_index(
                self.STYPE[train_layer_size:], required_grad=False)
            BEXP = torch.concat([first_ten_layer_BEXP, other_layer_BEXP])
            SMCMAX = torch.concat([first_ten_layer_SMCMAX, other_layer_SMCMAX])
            DKSAT = torch.concat([first_ten_layer_DKSAT, other_layer_DKSAT])
            PSISAT = torch.concat([first_ten_layer_PSISAT, other_layer_PSISAT])
            QUARTZ = self.Soil_Parameter.get_QTZ_by_index(self.STYPE)
        else:
            BEXP, SMCMAX, DKSAT, PSISAT, QUARTZ = self.Soil_Parameter.get_all_by_index(self.STYPE)
        F1 = torch.log10(PSISAT) + BEXP * torch.log10(SMCMAX) + 2.0
        REFSMC1 = SMCMAX * (5.79E-9 / DKSAT) ** (1 / (2 * BEXP + 3))
        SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
        WLTSMC1 = SMCMAX * (200. / PSISAT) ** (-1. / BEXP)
        DWSAT = BEXP * DKSAT * PSISAT / SMCMAX
        SMCWLT = 0.5 * WLTSMC1
        SMCDRY = SMCWLT
        return PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ


    @torch.jit.export
    def SFLX(self, FFROZP, LWDN, SOLDN, SOLNET, SFCPRS, PRCP, SFCTMP, Q2,
             TH2, Q2SAT, DQSDT2, SHDFAC,
             ALB, SNOALB, CMC, T1, STC, SMC, SH2O, SNOWH, SNEQV, CH
             , PC, XLAI, SNOTIME1, RIBB, ktime: int,
             lstm_input):
        """
        NOAH Physical Process Driver
        :param FFROZP:
        :param LWDN:
        :param SOLDN:
        :param SOLNET:
        :param SFCPRS:
        :param PRCP:
        :param SFCTMP:
        :param Q2:
        :param TH2:
        :param Q2SAT:
        :param DQSDT2:
        :param SHDFAC:
        :param ALB:
        :param SNOALB:
        :param CMC:
        :param T1:
        :param STC:
        :param SMC:
        :param SH2O:
        :param SNOWH:
        :param SNEQV:
        :param CH:
        :param PC:
        :param XLAI:
        :param SNOTIME1:
        :param RIBB:
        :return:
        """
        """global parameter"""
        self.k_time = ktime

        DKSAT, SMCMAX, SMCREF, SMCDRY, QUARTZ = self.read_surface_param()
        KDT = self.REFKDT * DKSAT / self.REFDK
        FRZFACT = (SMCMAX / SMCREF) * (0.412 / 0.468)
        FRZX = self.FRZK * FRZFACT

        SHDFAC = torch.where(torch.eq(tensor(self.VEGTYP), tensor(self.BARE)), tensor(0.0), SHDFAC)

        condition_1 = torch.ge(SHDFAC, self.SHDMAX)
        condition_2 = torch.le(SHDFAC, self.SHDMIN)
        INTERP_FRACTION = torch.clip((SHDFAC - self.SHDMIN) / (self.SHDMAX - self.SHDMIN), 0.0, 1.0)
        EMBRD = torch.where(condition_1, self.EMISSMAX, torch.where(condition_2, self.EMISSMIN, (
                1.0 - INTERP_FRACTION) * self.EMISSMIN + INTERP_FRACTION * self.EMISSMAX))
        XLAI = torch.where(condition_1, self.LAIMAX,
                           torch.where(condition_2, self.LAIMIN, torch.where(torch.logical_not(self.RDLAI2D), (
                                   1.0 - INTERP_FRACTION) * self.LAIMIN + INTERP_FRACTION * self.LAIMAX, XLAI)))
        ALB = torch.where(condition_1, self.ALBEDOMIN,
                          torch.where(condition_2, self.ALBEDOMAX,
                                      torch.where(torch.logical_not(self.USEMONALB), (
                                              1.0 - INTERP_FRACTION) * self.ALBEDOMAX + INTERP_FRACTION * self.ALBEDOMIN,
                                                  ALB)))
        Z0BRD = torch.where(condition_1, self.Z0MAX,
                            torch.where(condition_2, self.Z0MIN,
                                        (1.0 - INTERP_FRACTION) * self.Z0MIN + INTERP_FRACTION * self.Z0MAX))

        # ----------------------------------------------------------------------
        # Initialize the precipitation logical variable
        # ----------------------------------------------------------------------
        SNOWNG = torch.where(torch.logical_and(torch.gt(PRCP, 0.0), torch.gt(FFROZP, 0.5)), tensor(True),
                             torch.tensor(False))
        FRZGRA = torch.where(torch.logical_and(torch.gt(PRCP, 0.0), torch.le(T1, self.TFREEZ)), tensor(True),
                             torch.tensor(False))

        if SNEQV <= 1.0e-7:
            SNEQV = torch.tensor(0.0)  # snow water equivalent
            SNDENS = torch.tensor(0.0)
            SNOWH = torch.tensor(0.0)  # snow depth
            SNCOND = torch.tensor(1.0)
        else:
            SNDENS = SNEQV / SNOWH  # snow density
            if SNDENS > 1.0:
                raise ValueError('Physical snow depth is less than snow water equiv.')
            SNCOND = self.CSNOW(SNDENS)



        if SNOWNG or FRZGRA:
            SN_NEW = PRCP * self.DT * 0.001  # from kg/m^2*s convert to m
            SNEQV = SNEQV + SN_NEW  # snow water equivalent
            PRCPF = tensor(0.0)
            SNDENS, SNOWH = self.SNOW_NEW(SFCTMP, SN_NEW, SNOWH, SNDENS)
            SNCOND = self.CSNOW(SNDENS)
        else:
            PRCPF = PRCP

        DSOIL = -(0.5 * self.ZSOIL[0])
        DF1 = self.TDFCND_C05_Tensor(SMC[0].detach(), QUARTZ, SMCMAX, SH2O[0].detach(), self.STYPE[0])
        if SNEQV == 0.0:
            SNCOVR = tensor(0.0)
            ALBEDO = ALB
            EMISSI = EMBRD
            SSOIL = DF1 * (T1 - STC[0]) / DSOIL
            SNOTIME1 = tensor(0.0)
        else:
            SNCOVR = self.SNFRAC(SNEQV, self.SNUP, self.SALP)
            SNCOVR = torch.clamp(SNCOVR, min=0.0, max=1.0)
            ALBEDO, EMISSI, SNOTIME1 = self.ALCALC(ALB, SNOALB, EMBRD, SNCOVR, SNOWNG, SNOTIME1)
            DF1 = torch.where(torch.gt(SNCOVR, 0.97), SNCOND, DF1 * torch.exp(self.SBETA * SHDFAC))
            DTOT = SNOWH + DSOIL
            DF1A = SNOWH / DTOT * SNCOND + DSOIL / DTOT * DF1
            DF1 = DF1A * SNCOVR + DF1 * (1.0 - SNCOVR)
            SSOIL = DF1 * (T1 - STC[0]) / DTOT

        Z0 = torch.where(torch.gt(SNCOVR, 0.), self.SNOWZ0(SNCOVR, Z0BRD, SNOWH), Z0BRD)

        FDOWN = SOLNET + LWDN
        T2V = SFCTMP * (1.0 + 0.61 * Q2)
        # ETP: Potential Evapotranspiration
        EPSCA, ETP, RCH, RR, FLX2, T24 = self.PENMAN(SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT,
                                                     SNOWNG, FRZGRA, DQSDT2, EMISSI, SNCOVR)

        if SHDFAC > 0.:
            # canopy resistance and plant coefficient
            PC, RC, RCS, RCT, RCQ, RCSOIL = self.CANRES(SOLDN, CH, SFCTMP, Q2, SFCPRS, SH2O,
                                                        Q2SAT, DQSDT2, XLAI, EMISSI)

        if SNEQV == 0.0:
            STC, SMC, SH2O, DEW, DRIP, EC, EDIR, ETA, ET, ETT, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1, FLX1, FLX3 = self.NOPAC(
                ETP, PRCP, SMC, CMC, SHDFAC,
                SFCTMP, T24, TH2, FDOWN, EMISSI, STC, EPSCA, PC, RCH, RR, SH2O, KDT, FRZX, lstm_input)
            ETA_KINEMATIC = ETA
        else:
            (STC, SMC, SH2O, DEW, DRIP, ETP, EC, EDIR, ET, ETT, ETNS, ESNOW, FLX1, FLX3, RUNOFF1, RUNOFF2, RUNOFF3,
             SSOIL, SNOMLT, CMC, BETA, SNEQV, SNOWH, SNCOVR, SNDENS, T1) = self.SNOPAC(ETP, PRCP, PRCPF, SNOWNG,
                                                                                       SMC, CMC
                                                                                       , DF1, T1, SFCTMP, T24,
                                                                                       TH2, FDOWN, STC, PC, RCH, RR,
                                                                                       SNCOVR, SNEQV, SNDENS,
                                                                                       SNOWH,
                                                                                       SH2O, KDT, FRZX,
                                                                                       SHDFAC, FLX2, EMISSI, RIBB,
                                                                                       lstm_input)
            ETA_KINEMATIC = ESNOW + ETNS
        # ----------------------------------------------------------------------
        #   PREPARE SENSIBLE HEAT (H) FOR RETURN TO PARENT MODEL
        # ----------------------------------------------------------------------
        Q1 = Q2 + ETA_KINEMATIC * self.CP / RCH
        return (CMC.detach(), T1.detach(), STC, SMC,
                SH2O, SNOWH.detach(), SNEQV.detach(), SNOTIME1.detach(), ALBEDO.detach(),
                PC.detach(), Q1.detach(), Z0.detach(), Z0BRD.detach(), EMISSI.detach())

    # def EVAPO_LSTM(lstm_input, ETP1, NSOIL, NROOT, SMC, RTDIS, STYPE):
    #     global hc
    #     condition1 = ETP1 > 0.0
    #     condition2 = SHDFAC < 1.0
    #     condition3 = SHDFAC > 0.0
    #     condition4 = CMC > 0.0
    #     lstm_input = torch.cat(lstm_input, ETP1)
    #     out, hc = lstm_model(lstm_input, hc)  # out: 'EDIR', 'EC', 'ETT'
    #     EDIR, EC, ETT = out  # 将lstm模型输出为分解为三个变量，分别为表层蒸发，冠层蒸发以及植被蒸散发
    #     ET = torch.zeros(NSOIL)
    #     PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE[:NROOT], list(range(NROOT)))
    #     GX = torch.clamp((SMC[:NROOT] - SMCWLT) / (SMCREF - SMCWLT), min=0.0, max=1.0)
    #     SGX = torch.sum(GX)
    #     SGX = SGX / NROOT
    #     RTX = RTDIS[:NROOT] + GX[:NROOT] - SGX
    #     GX = GX * torch.clamp(RTX, min=0.0)
    #     DENOM = torch.sum(GX)
    #     ET[:NROOT] = ETT * GX / DENOM
    #     return ETA, EDIR, ETT, EC, ET

    def DEVAP(self, ETP1, SMC, SHDFAC) -> Tensor:
        """
        CALCULATE DIRECT SOIL EVAPORATION
        :param ETP1: Potential evapotranspiration
        :param SMC: SOIL MOISTURE
        :param SHDFAC: Areal fractional coverage of green vegetation
        :return:
            EDIR: Surface soil layer direct soil evaporation
        """
        DKSAT, SMCMAX, SMCREF, SMCDRY, QUARTZ = self.read_surface_param()
        SRATIO = (SMC - SMCDRY) / (SMCMAX - SMCDRY)
        SRATIO = torch.clamp(SRATIO, min=1e-9, max=None)  # Prevents numerical instability
        FX = torch.where(torch.gt(SRATIO, 0.0), torch.clamp(SRATIO ** self.FXEXP, min=0.0, max=1.0), tensor(0.0))
        return FX * (1.0 - SHDFAC) * ETP1

    def TRANSP(self, ETP1, SMC, CMC, SHDFAC, PC) -> Tensor:
        """
        CALCULATE TRANSPIRATION FOR THE VEG CLASS.
        :param ETP1:
        :param SMC:
        :param CMC:
        :param SHDFAC:
        :param PC:
        :return:
            ET
        """
        exponent_part = torch.clamp(CMC / self.CMCMAX, min=1e-9, max=None)
        ETP1A = torch.where(CMC != 0.0, SHDFAC * PC * ETP1 * (1.0 - exponent_part ** self.CFACTR), SHDFAC * PC * ETP1)
        ET = torch.zeros_like(SMC)
        SMCWLT, SMCREF = self.read_root_param()
        GX = torch.clamp((SMC[:self.NROOT] - SMCWLT) / (SMCREF - SMCWLT), min=0.0, max=1.0)
        SGX = torch.sum(GX)
        SGX = SGX / self.NROOT
        RTX = self.RTDIS[:self.NROOT] + GX[:self.NROOT] - SGX
        GX = GX * torch.clamp(RTX, min=0.0)
        DENOM = torch.sum(GX)
        DENOM = torch.where(torch.le(DENOM, 0.0), tensor(1), DENOM)
        ET[:self.NROOT] = ETP1A * GX / DENOM
        return ET

    def EVAPO(self, SMC, CMC, ETP1, SH2O, PC, SHDFAC):
        """
        CALCULATE SOIL MOISTURE FLUX.  THE SOIL MOISTURE CONTENT (SMC - A PER
        UNIT VOLUME MEASUREMENT) IS A DEPENDENT VARIABLE THAT IS UPDATED WITH
        PROGNOSTIC EQNS. THE CANOPY MOISTURE CONTENT (CMC) IS ALSO UPDATED.
        FROZEN GROUND VERSION:  NEW STATES ADDED: SH2O, AND FROZEN GROUND
        CORRECTION FACTOR, FRZFACT AND PARAMETER SLOPE.

        :param SMC:
        :param CMC:
        :param ETP1:
        :param SH2O:
        :param PC:
        :param SHDFAC:
        :return:
            tuple: A tuple containing:
                - ETA1:
                - EDIR:
                - ETT:
                - EC:
                - ET:
        """
        condition1 = ETP1 > 0.0
        condition2 = SHDFAC < 1.0
        condition3 = SHDFAC > 0.0
        condition4 = CMC > 0.0
        # surface evaporation
        # FXEXP veg parameter: bare soil evaporation exponent

        EDIR = torch.where(condition1 & condition2, self.DEVAP(ETP1, SMC[0], SHDFAC), tensor(0))

        ET = torch.where(condition1 & condition3,
                         self.TRANSP(ETP1, SH2O, CMC, SHDFAC, PC),
                         torch.zeros_like(SH2O))
        # total transpiration
        ETT = torch.sum(ET)
        # Constraints on exponential operations
        exponent_part = torch.clamp(CMC / self.CMCMAX, min=1e-9, max=None)
        # canopy evaporation
        EC = torch.where(condition1 & condition3 & condition4, SHDFAC * (exponent_part ** self.CFACTR) * ETP1,
                         tensor(0.0))
        CMC2MS = CMC / self.DT
        EC = torch.min(CMC2MS, EC)
        # total evapotranspiration
        ETA1 = EDIR + ETT + EC
        return ETA1, EDIR, ETT, EC, ET

    def WDFCND_NY06(self, SMC, SMCMAX, BEXP, DKSAT, DWSAT, SICE):
        """
        CALCULATE SOIL WATER DIFFUSIVITY AND SOIL HYDRAULIC CONDUCTIVITY.
        THE IMPEDANCE EFFECT OF UNDERGROUND ICE ON THE MOVEMENT OF LIQUID
        WATER IS CONSIDERED based ON THE METHODOLOGY DESCRIBED IN Wu X, Nan Z, Zhao S, et al.
        Spatial modeling of permafrost distribution and properties on the Qinghai‐Tibet Plateau[J].
        Permafrost and Periglacial Processes, 2018, 29(2): 86-99. doi: 10.1002/ppp.1971.
        :param SMC:
        :param SMCMAX:
        :param BEXP:
        :param DWSAT:
        :param DKSAT:
        :param SICE:
        :return: WDF, WCND
        """

        FICE = torch.min(torch.tensor(1.0), SICE / SMCMAX)
        FCR = torch.clamp(torch.exp(-4 * (1.0 - FICE)) - torch.exp(tensor(-4)), min=0.0) / (1.0 - torch.exp(tensor(-4)))

        EI = 1.25 * (DWSAT - 3) ** 2 + 6
        FACTR = torch.max(torch.tensor(0.01), SMC / SMCMAX)
        EXPON = BEXP + 2.0
        # water diffusivity
        WDF = DWSAT * FACTR ** EXPON
        WDF = torch.pow(10, -EI * SICE) * WDF

        EI = 1.25 * (DKSAT - 3) ** 2 + 6
        # hydraulic conductivity
        EXPON = 2.0 * BEXP + 3.0
        WCND = DKSAT * FACTR ** EXPON
        WCND = torch.pow(10, -EI * SICE) * WCND

        return WDF, WCND

    def FAC2MIT(self, SMCMAX: torch.Tensor) -> torch.Tensor:
        FLIMIT = torch.tensor(0.90)

        if torch.eq(SMCMAX, torch.tensor(0.395)):
            FLIMIT = torch.tensor(0.59)
        elif torch.isin(SMCMAX, torch.tensor([0.434, 0.404])):
            FLIMIT = torch.tensor(0.85)
        elif torch.isin(SMCMAX, torch.tensor([0.465, 0.406])):
            FLIMIT = torch.tensor(0.86)
        elif torch.isin(SMCMAX, torch.tensor([0.476, 0.439])):
            FLIMIT = torch.tensor(0.74)
        elif torch.isin(SMCMAX, torch.tensor([0.200, 0.464])):
            FLIMIT = torch.tensor(0.80)

        return FLIMIT

    def SRT(self, EDIR, ET, SH2O, SH2OA, PCPDRP, KDT, SICE, CMC, RHSCT):
        """
        CALCULATE THE RIGHT HAND SIDE OF THE TIME TENDENCY TERM OF THE SOIL
        WATER DIFFUSION EQUATION.  ALSO TO COMPUTE ( PREPARE ) THE MATRIX
        COEFFICIENTS FOR THE TRI-DIAGONAL MATRIX OF THE IMPLICIT TIME SCHEME.
        :param EDIR:
        :param ET:
        :param SH2O:
        :param SH2OA:
        :param PCPDRP:
        :param KDT:
        :param SICE:
        :return: RUNOFF1, RUNOFF2, AI, BI, CI, RHSTT
        """
        # OPT_INF = 2
        NSOIL = SH2O.size(0)
        # SICEMAX = torch.max(SICE)
        # PDDUM = PCPDRP
        # RUNOFF1 = tensor(0.0)
        RUNOFF2 = tensor(0.0)

        PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = self.read_all_param()
        NROOT = 8
        SMCAV = SMCMAX[0] - SMCWLT[0]
        DMAX = self.DENOM2[:NROOT] * SMCAV * (1 - (SH2OA[:NROOT] + SICE[:NROOT] - SMCWLT[:NROOT]) / SMCAV)
        DD = torch.sum(DMAX)
        DDT = DD * (1.0 - torch.exp(- KDT * self.DT / 86400.0))
        PX = torch.clamp(PCPDRP * self.DT, min=0.0)

        WDF2, WCND2 = self.WDFCND_NY06(SH2OA + SICE, SMCMAX, BEXP, DKSAT, DWSAT, SICE)

        INFMAX = torch.clamp((PX * (DDT / (PX + DDT))) / self.DT, min=WCND2[0], max=PX / self.DT)
        RUNOFF1 = torch.where(torch.gt(PCPDRP, INFMAX), PCPDRP - INFMAX, tensor(0.0))
        PDDUM = torch.where(torch.gt(PCPDRP, INFMAX), INFMAX, PCPDRP)

        #####################################################

        # Vector operations are used after optimization
        # DDZ2 = torch.zeros(NSOIL - 1)
        # DDZ2[0] = (0 - self.ZSOIL[1]) * 0.5
        # DDZ2[1:] = (self.ZSOIL[:-2] - self.ZSOIL[2:]) * 0.5
        AI = torch.zeros(NSOIL)
        CI = torch.zeros(NSOIL)
        RHSTT = torch.zeros(NSOIL)
        SLOPE_tensor = torch.ones(NSOIL - 1)
        delta_H2O = torch.zeros(NSOIL)
        SLOPE_tensor[-1] = self.SLOPE
        delta_H2O[:-1] = (SH2O[:-1] - SH2O[1:])

        AI[1:] = -WDF2[:-1] / (self.DDZ2 * self.DENOM2[1:])
        CI[:-1] = -WDF2[:-1] / (self.DDZ2 * self.DENOM2[:-1])
        BI = -(AI + CI)
        RHSTT[0] = CI[0] * delta_H2O[0] + (WCND2[0] - PDDUM + EDIR + ET[0]) / (-self.DENOM2[0])
        RHSTT[1:] = -AI[1:] * delta_H2O[:-1] + CI[1:] * delta_H2O[1:] + (
                WCND2[1:] * SLOPE_tensor - WCND2[:-1] + ET[1:]) / (
                        -self.DENOM2[1:])
        ######################################################
        AI = AI.detach()
        BI = BI.detach()
        CI = CI.detach()

        RHSTT = RHSTT * self.DT
        AI = AI * self.DT
        BI = 1 + BI * self.DT
        CI = CI * self.DT

        CO_Matrix = torch.diag(AI[1:], -1) + torch.diag(BI) + torch.diag(CI[:-1], 1)
        P = torch.linalg.solve(CO_Matrix, RHSTT)
        ###########################################################
        SMCMAX = SMCMAX.detach()
        PLUS = SMCMAX - (SH2O + P + SICE)
        if (PLUS < 0).any():
            WPLUS = torch.zeros(NSOIL + 1)
            for K in range(1, NSOIL + 1):
                WPLUS[K] = torch.clamp_min_(
                    ((SH2O[K - 1] + P[K - 1] + SICE[K - 1] + WPLUS[K - 1] / self.DENOM2[K - 1]) - SMCMAX[K - 1]) *
                    self.DENOM2[K - 1],
                    0)
            SH2OOUT = torch.clamp(SH2O + P + WPLUS[:-1] / self.DENOM2, min=tensor(0.0), max=SMCMAX - SICE)
            SMC = torch.clamp(SH2O + P + WPLUS[:-1] / self.DENOM2 + SICE, tensor(0.02), SMCMAX)
            RUNOFF3 = WPLUS[-1]
        else:
            SH2OOUT = torch.clamp(SH2O + P, min=tensor(0.0), max=SMCMAX - SICE)
            SMC = torch.clamp(SH2O + P + SICE, tensor(0.02), SMCMAX)
            RUNOFF3 = tensor(0.0)
        ##########################
        CMC = CMC + self.DT * RHSCT
        CMC = torch.where(CMC < 1E-20, tensor(0.0), CMC)
        CMC = torch.clamp(CMC, tensor(0.0), self.CMCMAX)
        SMC = SMC.detach()

        return RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMC, CMC

    def CSNOW(self, DSNOW):
        """
        CALCULATE SNOW TERMAL CONDUCTIVITY
        :param DSNOW: Snow density
        :return:
            SNOW TERMAL CONDUCTIVITY
        """
        UNIT = 0.11631
        C = 0.328 * 10 ** (2.25 * DSNOW)
        return 2.0 * UNIT * C

    def SNFRAC(self, SNEQV, SNUP, SALP):
        """
        CALCULATE SNOW FRACTION (0 -> 1)
        :param SNEQV:
        :param SNUP:
        :param SALP:
        :return:
            SNCOVR
        """
        RSNOW = SNEQV / SNUP
        SNCOVR = torch.where(torch.lt(SNEQV, SNUP),
                             1 - (torch.exp(- SALP * RSNOW) - RSNOW * torch.exp(- SALP)),
                             tensor(1.0))
        return SNCOVR

    def ALCALC(self, ALB, SNOALB, EMBRD, SNCOVR, SNOWNG, SNOTIME1):
        """
        CALCULATE ALBEDO INCLUDING SNOW EFFECT (0 -> 1)
        :param ALB: SNOWFREE ALBEDO
        :param SNOALB: MAXIMUM (DEEP) SNOW ALBEDO
        :param EMBRD: BACKGROUND EMISSIVITY
        :param SNCOVR: FRACTIONAL SNOW COVER
        :param SNOWNG: SNOW FLAG
        :param SNOTIME1: SNOW
        :return:
            ALBEDO, EMISSI, SNOTIME1
        """
        SNACCA = 0.94
        SNACCB = 0.58
        EMISSI = EMBRD + SNCOVR * (self.EMISSI_S - EMBRD)
        SNOALB1 = SNOALB + self.LVCOEF * (0.85 - SNOALB)
        SNOTIME1 = torch.where(SNOWNG, tensor(0.0), SNOTIME1 + self.DT)
        SNOALB2 = torch.where(SNOWNG, SNOALB1, SNOALB1 * (SNACCA ** ((SNOTIME1 / 86400.0) ** SNACCB)))
        SNOALB2 = torch.max(SNOALB2, ALB)
        ALBEDO = ALB + SNCOVR * (SNOALB2 - ALB)
        ALBEDO = torch.clamp(ALBEDO, max=SNOALB2)
        return ALBEDO, EMISSI, SNOTIME1

    def TDFCND_C05_Tensor(self, SMC, QZ, SMCMAX, SH2O, NSOILTYPE):
        """
        input data is tensor
        :param SMC:
        :param QZ:
        :param SMCMAX:
        :param SH2O:
        :param NSOILTYPE:
        :return:
        """
        THKICE = 2.2
        THKW = 0.57
        THKO = 2.0
        THKQTZ = 7.7
        SATRATIO = SMC / SMCMAX
        THKS = (THKQTZ ** QZ) * (THKO ** (1.0 - QZ))
        THKS = torch.clamp(THKS, min=1e-9, max=None)
        XUNFROZ = SH2O / SMC
        XU = XUNFROZ * SMCMAX
        THKSAT = THKS ** (1.0 - SMCMAX) * THKICE ** (SMCMAX - XU) * THKW ** XU
        GRAVELBEDROCKOTHER = torch.full_like(NSOILTYPE, 3)  # default value: 3
        GRAVELBEDROCKOTHER = torch.where(torch.eq(NSOILTYPE, 13), tensor(1), GRAVELBEDROCKOTHER)
        GRAVELBEDROCKOTHER = torch.where(torch.eq(NSOILTYPE, 14), tensor(2), GRAVELBEDROCKOTHER)
        THKDRY = torch.where(torch.eq(GRAVELBEDROCKOTHER, 1), 1.70 * 10 ** (-1.80 * SMCMAX),
                             torch.where(
                                 torch.eq(GRAVELBEDROCKOTHER, 2),
                                 0.039 * SMCMAX ** (-2.2),
                                 (0.135 * (1.0 - SMCMAX) * 2700.0 + 64.7) / (2700.0 - 0.947 * (1.0 - SMCMAX) * 2700.0))
                             )
        AKE = torch.where((SH2O + 0.0005) < SMC,
                          torch.where(torch.eq(GRAVELBEDROCKOTHER, 1),
                                      (1.7 * SATRATIO) / (1 + (1.7 - 1) * SATRATIO),
                                      SATRATIO),
                          torch.where(torch.eq(GRAVELBEDROCKOTHER, 1),
                                      (4.6 * SATRATIO) / (1 + (4.6 - 1) * SATRATIO),
                                      torch.where(SATRATIO > 0.1, torch.log10(SATRATIO) + 1.0, tensor(0.0)))
                          )
        DF = AKE * (THKSAT - THKDRY) + THKDRY
        return DF

    def SNOW_NEW(self, TEMP, NEWSN, SNOWH, SNDENS):
        """
        CALCULATE SNOW DEPTH AND DENSITY TO ACCOUNT FOR THE NEW SNOWFALL.
        NEW VALUES OF SNOW DEPTH & DENSITY RETURNED.
        :param TEMP:
        :param NEWSN:
        :param SNOWH:
        :param SNDENS:
        :return:
            SNDENS, SNOWH
        """
        SNOWHC = SNOWH * 100.
        NEWSNC = NEWSN * 100.
        TEMPC = TEMP - 273.15
        DSNEW = torch.where(torch.le(TEMPC, -15.), tensor(0.05), 0.05 + 0.0017 * (TEMPC + 15.) ** 1.5)
        HNEWC = NEWSNC / DSNEW
        SNDENS = torch.where(SNOWHC + HNEWC < 1.0E-3, torch.max(DSNEW, SNDENS),
                             (SNOWHC * SNDENS + HNEWC * DSNEW) / (SNOWHC + HNEWC))
        SNOWHC = SNOWHC + HNEWC
        SNOWH = SNOWHC * 0.01
        return SNDENS, SNOWH

    def SNOWZ0(self, SNCOVR, Z0BRD, SNOWH):
        """
        CALCULATE TOTAL ROUGHNESS LENGTH OVER SNOW SNCOVR FRACTIONAL SNOW COVER
        :param SNCOVR:
        :param Z0BRD:
        :param SNOWH:
        :return:
            Z0
        """
        Z0S = 0.001
        BURIAL = 7 * Z0BRD - SNOWH
        Z0EFF = torch.where(BURIAL < 0.0007, Z0S, BURIAL / 7.0)
        return (1. - SNCOVR) * Z0BRD + SNCOVR * Z0EFF

    def WDFCND(self, SMC, SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX):
        # Calculate the ratio of the actual to the max possible soil water content
        FACTR1 = 0.05 / SMCMAX
        FACTR2 = SMC / SMCMAX
        FACTR1 = torch.min(FACTR1, FACTR2)

        # Prepare an exponential coefficient and calculate the soil water diffusivity
        EXPON = BEXP + 2.0
        WDF = DWSAT * torch.pow(FACTR2, EXPON)

        if SICEMAX > 0.0:
            VKWGT = 1.0 / (1.0 + (500.0 * SICEMAX) ** 3.0)
            WDF = VKWGT * WDF + (1.0 - VKWGT) * DWSAT * torch.pow(FACTR1, EXPON)

        # Reset the exponential coefficient and calculate the hydraulic conductivity
        EXPON = (2.0 * BEXP) + 3.0
        WCND = DKSAT * torch.pow(FACTR2, EXPON)

        return WDF, WCND

    def TMPAVG(self, TUP, TM, TDN):
        """
        CALCULATE SOIL LAYER AVERAGE TEMPERATURE (TAVG) IN FREEZING/THAWING
        LAYER USING UP, DOWN, AND MIDDLE LAYER TEMPERATURES (TUP, TDN, TM),
        WHERE TUP IS AT TOP BOUNDARY OF LAYER, TDN IS AT BOTTOM BOUNDARY OF LAYER.
        TM IS LAYER PROGNOSTIC STATE TEMPERATURE.
        :param TUP:
        :param TM:
        :param TDN:
        :return: TSVG
        """
        T0 = 273.15

        DZ = torch.zeros_like(TUP)
        DZ[0] = -self.ZSOIL[0]
        DZ[1:] = self.ZSOIL[:-1] - self.ZSOIL[1:]

        DZH = DZ * 0.5
        X0 = (T0 - TM) * DZH / (TDN - TM)
        XUP_1 = (T0 - TUP) * DZH / (TM - TUP)
        XUP_2 = DZH - (T0 - TUP) * DZH / (TM - TUP)
        XDN_1 = DZH - (T0 - TM) * DZH / (TDN - TM)
        XDN_2 = (T0 - TM) * DZH / (TDN - TM)

        TAVG = torch.where(
            torch.lt(TUP, T0),
            torch.where(
                torch.lt(TM, T0),
                torch.where(
                    torch.lt(TDN, T0),
                    (TUP + 2.0 * TM + TDN) / 4.0,
                    0.5 * (TUP * DZH + TM * (DZH + X0) + T0 * (2.0 * DZH - X0)) / DZ
                ),
                torch.where(
                    torch.lt(TDN, T0),
                    0.5 * (TUP * XUP_1 + T0 * (2.0 * DZ - XUP_1 - XDN_2) + TDN * XDN_2) / DZ,
                    0.5 * (TUP * XUP_1 + T0 * (2.0 * DZ - XUP_1)) / DZ
                )
            ),
            torch.where(
                torch.lt(TM, T0),
                torch.where(
                    torch.lt(TDN, T0),
                    0.5 * (T0 * (DZ - XUP_2) + TM * (DZH + XUP_2) + TDN * DZH) / DZ,
                    0.5 * (T0 * (2.0 * DZ - XUP_2 - XDN_2) + TM * (XUP_2 + XDN_2)) / DZ
                ),
                torch.where(
                    torch.lt(TDN, T0),
                    (T0 * (DZ - XDN_1) + 0.5 * (T0 + TDN) * XDN_1) / DZ,
                    (TUP + 2.0 * TM + TDN) / 4.0
                )
            )
        )

        return TAVG

    def TBND(self, TU, TB):
        """
        CALCULATE TEMPERATURE ON THE BOUNDARY OF THE LAYER BY INTERPOLATION OF
        THE MIDDLE LAYER TEMPERATURES
        :param TU:
        :param TB:
        :return: TBAND
        """
        delta = torch.zeros_like(TU)
        delta[0] = (0 - self.ZSOIL[0]) / (0 - self.ZSOIL[1])
        delta[1:-1] = (self.ZSOIL[0:-2] - self.ZSOIL[1:-1]) / (self.ZSOIL[0:-2] - self.ZSOIL[2:])
        delta[-1] = (self.ZSOIL[-2] - self.ZSOIL[-1]) / (self.ZSOIL[-2] - (2 * self.ZBOT - self.ZSOIL[-1]))
        return TU + (TB - TU) * delta

    def FRH2O_tensor(self, TKELV, SMC, SH2O, SMCMAX, BEXP, PSIS):
        """
        CALCULATE AMOUNT OF SUPERCOOLED LIQUID SOIL WATER CONTENT IF
        TEMPERATURE IS BELOW 273.15K (T0).  REQUIRES NEWTON-TYPE ITERATION TO
        SOLVE THE NONLINEAR IMPLICIT EQUATION GIVEN IN EQN 17 OF KOREN ET AL
        (1999, JGR, VOL 104(D16), 19569-19585).
        :param TKELV:
        :param SMC:
        :param SH2O:
        :param SMCMAX:
        :param BEXP:
        :param PSIS:
        :return:
            FREE
        """
        CK = torch.full_like(SH2O, 8.0)
        BLIM = torch.full_like(SH2O, 5.5)
        ERROR = torch.full_like(SH2O, 0.005)
        HLICE = torch.full_like(SH2O, 3.335E5)
        GS = torch.full_like(SH2O, 9.81)
        T0 = torch.full_like(SH2O, 273.15)

        BX = torch.where(torch.gt(BEXP, BLIM), BLIM, BEXP)

        NLOG = 0

        KCOUNT = torch.zeros_like(SH2O)
        DF = torch.zeros_like(SH2O)
        DENOM = torch.zeros_like(SH2O)
        SWLK = torch.zeros_like(SH2O)
        DSWL = torch.zeros_like(SH2O)

        SWL = torch.where(SH2O < 0.02, SMC - 0.02, SMC - SH2O)
        SWL = torch.clamp(SWL, tensor(0.0))

        while True:
            if not (NLOG < 10 and torch.any(torch.eq(KCOUNT, 0))):
                break
            NLOG = NLOG + 1
            DF = torch.where(torch.eq(KCOUNT, 0) & torch.le(TKELV, T0 - 1.E-3), (
                    torch.log((PSIS * GS / HLICE) * ((1. + CK * SWL) ** 2.) * (SMCMAX / (SMC - SWL)) ** BX)
                    - torch.log(-(TKELV - T0) / TKELV)), DF)
            if torch.isnan(DF).any():
                print(f"error {self.file_name}")
            DENOM = torch.where(torch.eq(KCOUNT, 0), 2. * CK / (1. + CK * SWL) + BX / (SMC - SWL), DENOM)
            SWLK = torch.where(torch.eq(KCOUNT, 0), SWL - DF / DENOM, SWLK)
            SWLK = torch.where(torch.eq(KCOUNT, 0), torch.where(torch.gt(SWLK, SMC - 0.02), SMC - 0.02, SWLK), SWLK)
            SWLK = torch.clamp(SWLK, min=0.0)
            DSWL = torch.where(torch.eq(KCOUNT, 0), torch.abs(SWLK - SWL), DSWL)
            SWL = torch.where(torch.eq(KCOUNT, 0), SWLK, SWL)
            KCOUNT = torch.where(torch.eq(KCOUNT, 0), torch.where(torch.le(DSWL, ERROR), KCOUNT + 1, KCOUNT), KCOUNT)
        FREE = SMC - SWL
        if torch.any(torch.eq(KCOUNT, 0)):
            print('Flerchinger USED in NEW version. Iterations=', NLOG)
            FK = (((HLICE / (GS * (-PSIS))) * ((TKELV - T0) / TKELV)) ** (-1 / BX)) * SMCMAX
            FK = torch.max(FK, tensor(0.02))
            FREE_ = torch.min(FK, SMC)
            FREE = torch.where(torch.eq(KCOUNT, 0), FREE_, FREE)
        FREE = torch.where(torch.gt(TKELV, T0 - 1.E-3), SMC, FREE)
        return FREE

    def FRH2O_NY06(self, TKELV, SMCMAX, BEXP, PSIS, SMC):
        """
        CALCULATE AMOUNT OF SUPERCOOLED LIQUID SOIL WATER CONTENT IF
        TEMPERATURE IS BELOW 273.15K (T0).  REQUIRES NEWTON-TYPE ITERATION TO
        SOLVE THE NONLINEAR IMPLICIT EQUATION GIVEN IN EQN 3 OF NIU ET AL
        (NIU, JGR, VOL 104(D16), 19569-19585).
        :param TKELV:
        :param SMCMAX:
        :param BEXP:
        :param PSIS:
        :param SMC:
        :return:
             FREE
        """
        HFUS = 0.3336E06
        TFRZ = 273.16
        GRAV = 9.80616
        SMP = (HFUS * (TFRZ - TKELV)) / (GRAV * TKELV * PSIS)
        SMP = torch.clamp(SMP, min=1e-9)
        FK = SMCMAX * (SMP ** (-1. / BEXP))
        FREE = torch.max(torch.min(FK, SMC), tensor(0.))
        return FREE

    def SNOWPACK(self, ESD, SNOWH, SNDENS, TSNOW, TSOIL):
        """
        CALCULATE COMPACTION OF SNOWPACK UNDER CONDITIONS OF INCREASING SNOW
        DENSITY, AS OBTAINED FROM AN APPROXIMATE SOLUTION OF E. ANDERSON'S
        DIFFERENTIAL EQUATION (3.29), NOAA TECHNICAL REPORT NWS 19, BY VICTOR
        KOREN, 03/25/95.
        :param ESD:
        :param SNOWH:
        :param SNDENS:
        :param TSNOW:
        :param TSOIL:
        :return:
        """
        C1 = 0.01
        C2 = 21.0
        ESDC = ESD * 100.
        DTHR = self.DT / 3600.
        TSNOWC = TSNOW - 273.15
        TSOILC = TSOIL - 273.15
        TAVGC = 0.5 * (TSNOWC + TSOILC)
        ESDCX = torch.clamp_max(ESDC, 1.E-2)
        BFAC = DTHR * C1 * torch.exp(0.08 * TAVGC - C2 * SNDENS)
        PEXP = tensor(0.0)
        for J in range(4, 0, -1):
            PEXP = (1.0 + PEXP) * BFAC * ESDCX / (J + 1)
        PEXP = PEXP + 1.
        DSX = SNDENS * PEXP
        DSX = torch.clamp(DSX, min=0.05, max=0.4)
        SNDENS = torch.where(torch.ge(TSNOWC, 0),
                             torch.clamp(SNDENS * (1. - 0.13 * DTHR / 24.) + 0.13 * DTHR / 24., max=0.04), DSX)
        SNOWHC = ESDC / SNDENS
        SNOWH = SNOWHC * 0.01
        return SNOWH, SNDENS

    def SNKSRC(self, TAVG, SMC, SH2O, SMCMAX, PSISAT, BEXP, QTOT):
        """
        CALCULATE SINK/SOURCE TERM OF THE TERMAL DIFFUSION EQUATION. (SH2O) IS
        AVAILABLE LIQUED WATER.
        :param TAVG:
        :param SMC:
        :param SH2O:
        :param SMCMAX:
        :param PSISAT:
        :param BEXP:
        :param QTOT:
        :return:
            TSNSR, XH2O
        """
        DH2O = 1.0000E3
        HLICE = 3.3350E5
        DZ = torch.zeros_like(SH2O)
        DZ[0] = -self.ZSOIL[0]
        DZ[1:] = self.ZSOIL[:-1] - self.ZSOIL[1:]
        FREE = self.FRH2O_tensor(TAVG.detach(), SMC, SH2O, SMCMAX, BEXP, PSISAT)
        FREE = FREE - SH2O.detach() + SH2O
        XH2O = SH2O + QTOT.detach() * self.DT / (DH2O * HLICE * DZ)
        XH2O = torch.where(torch.lt(XH2O, SH2O) & torch.lt(XH2O, FREE), torch.min(FREE, SH2O), XH2O)
        XH2O = torch.where(torch.gt(XH2O, SH2O) & torch.gt(XH2O, FREE), torch.max(FREE, SH2O), XH2O)
        XH2O = torch.clamp(XH2O, tensor(0.0), SMC.detach())
        TSNSR = -DH2O * HLICE * DZ * (XH2O - SH2O).detach() / self.DT
        return TSNSR, XH2O

    def SHFLX(self, STC, SMC, YY, ZZ1, SH2O, DF1):
        """
        UPDATE THE TEMPERATURE STATE OF THE SOIL COLUMN BASED ON THE THERMAL
        DIFFUSION EQUATION AND UPDATE THE FROZEN SOIL MOISTURE CONTENT BASED
        ON THE TEMPERATURE.
        :param STC:
        :param SMC:
        :param YY:
        :param ZZ1:
        :param SH2O:
        :param DF1:
        :return:
        """
        OPT_TCND = 2
        NSOIL = SH2O.size(0)
        T0 = 273.15
        CAIR = 1004.0
        CICE = 2.106e6
        CH2O = 4.2e6
        TBK = 0.0
        TBK1 = 0.0
        CSOIL_LOC = self.CSOIL

        # Vector operations are used after optimization
        PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = self.read_all_param()
        DF1N = self.TDFCND_C05_Tensor(SMC.detach(), QUARTZ, SMCMAX, SH2O.detach(), self.STYPE)
        SICE = SMC - SH2O
        DF1N[0] = DF1
        # DENOM2 = torch.zeros(NSOIL)  # ΔZk
        # DDZ2 = torch.zeros(NSOIL - 1)  # ΔZk_tilde
        # DENOM2[0] = -ZSOIL[0]
        # DENOM2[1:] = ZSOIL[:-1] - ZSOIL[1:]
        # DDZ2[0] = (0 - ZSOIL[1]) * 0.5
        # DDZ2[1:] = (ZSOIL[:-2] - ZSOIL[2:]) * 0.5
        AI = torch.zeros(NSOIL)  # subdiagonal elements
        CI = torch.zeros(NSOIL)  # superdiagonal elements
        BI = torch.zeros(NSOIL)  # diagonal elements
        RHSTS = torch.zeros(NSOIL)  # Right hand
        delta_STC = torch.zeros(NSOIL)
        delta_STC[:-1] = (STC[:-1] - STC[1:])
        delta_STC[-1] = STC[-1] - self.TBOT
        # Calculation Graph of Separating Temperature and Soil Water
        HCPCT = SH2O.detach() * CH2O + (1.0 - SMCMAX) * CSOIL_LOC + (
                SMCMAX - SMC.detach()) * CAIR + SICE.detach() * CICE
        # HCPCT = SH2O * CH2O + (1.0 - SMCMAX) * CSOIL_LOC + (SMCMAX - SMC) * CAIR + SICE * CICE

        AI[1:] = -DF1N[:-1] / (self.DENOM2[1:] * self.DDZ2 * HCPCT[1:])
        CI[:-1] = -DF1N[:-1] / (self.DENOM2[:-1] * self.DDZ2 * HCPCT[:-1])
        BI[0] = -CI[0] + DF1N[0] / (0.5 * self.ZSOIL[0] * self.ZSOIL[0] * HCPCT[0] * ZZ1)
        BI[1:] = -(AI[1:] + CI[1:])
        SSOIL = DF1N[0] * (STC[0] - YY) / (0.5 * self.ZSOIL[0] * ZZ1)
        RHSTS[0] = (SSOIL - delta_STC[0] * DF1N[0] / self.DDZ2[0]) / (self.DENOM2[0] * HCPCT[0])
        RHSTS[1:-1] = (-AI[1:-1] * delta_STC[:-2] + CI[1:-1] * delta_STC[1:-1])
        RHSTS[-1] = -AI[-1] * delta_STC[-2] - (DF1N[-1] * delta_STC[-1]) / (
                self.DENOM2[-1] * HCPCT[-1] * (.5 * (self.ZSOIL[-2] + self.ZSOIL[-1]) - self.ZBOT))
        QTOT = RHSTS * self.DENOM2 * HCPCT
        TB = torch.zeros(NSOIL)
        TB[:-1] = STC[1:]
        TB[-1] = self.TBOT
        TDN = self.TBND(STC, TB)
        TUP = torch.zeros(NSOIL)
        TUP[0] = (YY + (ZZ1 - 1) * STC[0]) / ZZ1  # TSURF
        TUP[1:] = TDN[:-1]
        TAVG = self.TMPAVG(TUP, STC, TDN)  # Average Temperature
        TSNSR, SH2O_New = self.SNKSRC(TAVG, SMC, SH2O, SMCMAX, PSISAT, BEXP, QTOT)  # phase change
        condition = torch.gt(SICE, 0) | torch.lt(STC, T0) | torch.lt(TUP, T0) | torch.lt(TDN,
                                                                                         T0)  # Phase change and calculate supercooled liquid water
        TSNSR = torch.where(condition, TSNSR, 0)  # The diffusion equation right-hand term source and sink
        SH2O_New = torch.where(condition, SH2O_New, SH2O)
        RHSTS = RHSTS + TSNSR / (self.DENOM2 * HCPCT)

        RHSTS = RHSTS * self.DT
        AI = AI * self.DT
        BI = 1 + BI * self.DT
        CI = CI * self.DT

        CO_Matrix = torch.diag(AI[1:], -1) + torch.diag(BI) + torch.diag(CI[:-1], 1)
        P = torch.linalg.solve(CO_Matrix, RHSTS)
        STC = STC + P

        T1 = (YY + (ZZ1 - 1.0) * STC[0]) / ZZ1
        SSOIL = DF1 * (STC[0] - T1) / (0.5 * self.ZSOIL[0])
        return STC, T1, SSOIL, SH2O_New

    def SMFLX(self, SMC, CMC, PRCP1, SH2O, KDT, SHDFAC, EDIR, EC, ET):
        """
        CALCULATE SOIL MOISTURE FLUX.  THE SOIL MOISTURE CONTENT (SMC - A PER
        UNIT VOLUME MEASUREMENT) IS A DEPENDENT VARIABLE THAT IS UPDATED WITH
        PROGNOSTIC EQNS. THE CANOPY MOISTURE CONTENT (CMC) IS ALSO UPDATED.
        FROZEN GROUND VERSION:  NEW STATES ADDED: SH2O, AND FROZEN GROUND
        CORRECTION FACTOR, FRZFACT AND PARAMETER SLOPE.
        :param SMC:
        :param CMC:
        :param PRCP1:
        :param SH2O:
        :param KDT:
        :param SHDFAC:
        :param EDIR:
        :param EC:
        :param ET:
        :return:
            SH2OOUT, SMCOUT, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP
        """
        RHSCT = SHDFAC * PRCP1 - EC
        # DRIP = tensor(0.0)
        # FAC2 = tensor(0.0)
        TRHSCT = self.DT * RHSCT
        EXCESS = CMC + TRHSCT
        DRIP = torch.where(torch.gt(EXCESS, self.CMCMAX), EXCESS - self.CMCMAX, tensor(0.0))

        PCPDRP = (1. - SHDFAC) * PRCP1 + DRIP / self.DT
        SICE = (SMC - SH2O)
        PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = self.read_all_param()
        FAC2 = torch.max(SH2O.detach() / SMCMAX)
        SMCMAX = SMCMAX[0]
        FLIMIT = self.FAC2MIT(SMCMAX)
        """Kalnay E, Kanamitsu M. Time Schemes for Strongly Nonlinear Damping Equations[J]. Monthly Weather Review, 1988, 116(10): 1945-1958."""
        if (PCPDRP * self.DT) > (0.0001 * 1000.0 * (- self.ZSOIL[0]) * SMCMAX) or FAC2 > FLIMIT:
            RUNOFF1, RUNOFF2, RUNOFF3, SH2OFG, SMCFG, DUMMY = self.SRT(EDIR, ET, SH2O, SH2O, PCPDRP,
                                                                       KDT, SICE, CMC, RHSCT)
            SH2OA = 0.5 * (SH2O.detach() + SH2OFG)
            RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMCOUT, CMC = self.SRT(EDIR, ET, SH2O, SH2OA, PCPDRP,
                                                                       KDT, SICE, CMC, RHSCT)
        else:
            RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMCOUT, CMC = self.SRT(EDIR, ET, SH2O, SH2O, PCPDRP,
                                                                       KDT, SICE, CMC, RHSCT)
        return SH2OOUT, SMCOUT, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP

    def PENMAN(self, SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT, SNOWNG, FRZGRA, DQSDT2,
               EMISSI_IN,
               SNCOVR):
        """
        CALCULATE POTENTIAL EVAPORATION FOR THE CURRENT POINT.  VARIOUS
        PARTIAL SUMS/PRODUCTS ARE ALSO CALCULATED AND PASSED BACK TO THE
        CALLING ROUTINE FOR LATER USE.
        :param SFCTMP:
        :param SFCPRS:
        :param CH:
        :param T2V:
        :param TH2:
        :param PRCP:
        :param FDOWN:
        :param SSOIL:
        :param Q2:
        :param Q2SAT:
        :param SNOWNG:
        :param FRZGRA:
        :param DQSDT2:
        :param EMISSI_IN:
        :param SNCOVR:
        :return:
            EPSCA, ETP, RCH, RR, FLX2, T24
        """
        CP = 1004.6
        CPH2O = 4.218E+3
        CPICE = 2.106E+3
        ELCP = 2.4888E+3
        LSUBF = 3.335E+5
        LSUBC = 2.501000E+6
        SIGMA = 5.67E-8
        LSUBS = 2.83E+6
        RD = 287.04

        EMISSI = EMISSI_IN
        ELCP1 = (1.0 - SNCOVR) * ELCP + SNCOVR * ELCP * LSUBS / LSUBC
        LVS = (1.0 - SNCOVR) * LSUBC + SNCOVR * LSUBS
        FLX2 = tensor(0.0)

        DELTA = ELCP1 * DQSDT2
        T24 = SFCTMP * SFCTMP * SFCTMP * SFCTMP
        RR = EMISSI * T24 * 6.48E-8 / (SFCPRS * CH) + 1.0
        RHO = SFCPRS / (RD * T2V)
        RCH = RHO * CP * CH

        con = torch.logical_and(~SNOWNG, PRCP > 0.0)
        RR = torch.where(con, RR + CPH2O * PRCP / RCH, RR + CPICE * PRCP / RCH)

        FNET = FDOWN - EMISSI * SIGMA * T24 - SSOIL.detach()
        FLX2 = torch.where(FRZGRA, -LSUBF * PRCP, FLX2)
        FNET = torch.where(FRZGRA, FNET - FLX2, FNET)

        RAD = FNET / RCH + TH2 - SFCTMP
        A = ELCP1 * (Q2SAT - Q2)
        EPSCA = (A * RR + RAD * DELTA) / (DELTA + RR)
        ETP = EPSCA * RCH / LVS
        return EPSCA, ETP, RCH, RR, FLX2, T24

    def CANRES(self, SOLAR, CH, SFCTMP, Q2, SFCPRS, SMC, Q2SAT, DQSDT2, XLAI, EMISSI):
        """
        CALCULATE CANOPY RESISTANCE WHICH DEPENDS ON INCOMING SOLAR RADIATION,
        AIR TEMPERATURE, ATMOSPHERIC WATER VAPOR PRESSURE DEFICIT AT THE
        LOWEST MODEL LEVEL, AND SOIL MOISTURE (PREFERABLY UNFROZEN SOIL
        MOISTURE RATHER THAN TOTAL)
        :param SOLAR: INCOMING SOLAR RADIATION
        :param CH: SURFACE EXCHANGE COEFFICIENT FOR HEAT AND MOISTURE
        :param SFCTMP: AIR TEMPERATURE AT 1ST LEVEL ABOVE GROUND
        :param Q2: AIR HUMIDITY AT 1ST LEVEL ABOVE GROUND
        :param SFCPRS: SURFACE PRESSURE
        :param SMC: VOLUMETRIC SOIL MOISTURE
        :param Q2SAT: SATURATION AIR HUMIDITY AT 1ST LEVEL ABOVE GROUND
        :param DQSDT2: SLOPE OF SATURATION HUMIDITY FUNCTION WRT TEMP
        :param EMISSI:  Emissivity ( fraction )
        :return:
            PC, RC, RCS, RCT, RCQ, RCSOIL
        """
        CP = 1004.5
        RD = 287.04
        SIGMA = 5.67E-8
        SLV = 2.501000E6

        FF = 0.55 * 2.0 * SOLAR / (self.RGL * XLAI)
        RCS = (FF + self.RSMIN / self.RSMAX) / (1.0 + FF)
        RCS = torch.clamp_min(RCS, 0.0001)

        RCT = 1.0 - 0.0016 * ((self.TOPT - SFCTMP) ** 2.0)
        RCT = torch.clamp_min(RCT, 0.0001)

        RCQ = 1.0 / (1.0 + self.HS * (Q2SAT - Q2))
        RCQ = torch.clamp_min(RCQ, 0.01)
        # SMCWLT, SMCREF
        SMCWLT, SMCREF = self.read_root_param()
        GX = torch.clamp((SMC[:self.NROOT] - SMCWLT) / (SMCREF - SMCWLT), min=0.0, max=1.0)
        delta = torch.zeros_like(self.ZSOIL)
        delta[0] = self.ZSOIL[0]
        delta[1:] = self.ZSOIL[1:] - self.ZSOIL[:-1]
        PART = delta[:self.NROOT] * GX / self.ZSOIL[self.NROOT - 1]

        RCSOIL = PART.sum()

        RCSOIL = torch.clamp_min(RCSOIL, 0.0001)

        RC = self.RSMIN / (XLAI * RCS * RCT * RCQ * RCSOIL)

        RR = (4. * EMISSI * SIGMA * RD / CP) * (SFCTMP ** 4.) / (SFCPRS * CH) + 1.0
        DELTA = (SLV / CP) * DQSDT2

        PC = (RR + DELTA) / (RR * (1. + RC * CH) + DELTA)
        return PC, RC, RCS, RCT, RCQ, RCSOIL

    def NOPAC(self, ETP, PRCP, SMC, CMC, SHDFAC,
              SFCTMP, T24, TH2, FDOWN, EMISSI, STC, EPSCA, PC, RCH, RR, SH2O, KDT, FRZFACT, lstm_input):
        """
        CALCULATE SOIL MOISTURE AND HEAT FLUX VALUES AND UPDATE SOIL MOISTURE
        CONTENT AND SOIL HEAT CONTENT VALUES FOR THE CASE WHEN NO SNOW PACK IS
        PRESENT.
        :param ETP:
        :param PRCP:
        :param SMC:
        :param CMC:
        :param SHDFAC:
        :param SFCTMP:
        :param T24:
        :param TH2:
        :param FDOWN:
        :param EMISSI:
        :param STC:
        :param EPSCA:
        :param PC:
        :param RCH:
        :param RR:
        :param SH2O:
        :param KDT:
        :param FRZFACT:
        :return:
            STC, SMC, SH2O, DEW, DRIP, EC, EDIR, ETA, ET, ETT, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1, FLX1, FLX3
        """
        CPH2O = 4.218E+3
        SIGMA = 5.67E-8
        PRCP1 = PRCP * 0.001
        ETP1 = ETP * 0.001
        # - EDIR: soil Evaporation
        # - ETT: total Transpiration
        # - EC: Canopy Evaporation
        # - ET: each layer Transpiration
        ETA, EDIR, ETT, EC, ET = self.EVAPO(SMC, CMC, ETP1, SH2O, PC, SHDFAC)
        DEW = torch.where(torch.gt(ETP, 0), tensor(0.0), -ETP1)
        PRCP1 = torch.where(torch.gt(ETP, 0), PRCP * 0.001, PRCP1 + DEW)
        SH2O, SMC, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP = self.SMFLX(SMC, CMC, PRCP1, SH2O,
                                                                     KDT, SHDFAC, EDIR, EC, ET)
        ETA = torch.where(torch.gt(ETP, 0), ETA * 1000, ETP)  # total evapotranspiration KG M-2 S-1
        BETA = torch.where(torch.gt(ETP, 0), ETA.detach() / ETP, torch.where(torch.lt(ETP, 0.0), 1, 0))
        EC = EC * 1000
        ET = ET * 1000
        EDIR = EDIR * 1000
        DKSAT, SMCMAX, SMCREF, SMCDRY, QUARTZ = self.read_surface_param()
        DF1 = self.TDFCND_C05_Tensor(SMC[0].detach(), QUARTZ, SMCMAX, SH2O[0].detach(), self.STYPE[0])
        DF1 = DF1 * torch.exp(self.SBETA * SHDFAC)
        YYNUM = FDOWN - EMISSI * SIGMA * T24
        YY = SFCTMP + (YYNUM / RCH + TH2 - SFCTMP - BETA * EPSCA) / RR
        ZZ1 = DF1 / (-0.5 * self.ZSOIL[0] * RCH * RR) + 1.0
        STC, T1, SSOIL, SH2O = self.SHFLX(STC, SMC, YY, ZZ1, SH2O, DF1)
        FLX1 = CPH2O * PRCP * (T1 - SFCTMP)
        FLX3 = tensor(0.0)
        return STC, SMC, SH2O, DEW, DRIP, EC, EDIR, ETA, ET, ETT, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1, FLX1, FLX3

    def SNOPAC(self, ETP, PRCP, PRCPF, SNOWNG, SMC, CMC, DF1,
               T1, SFCTMP, T24, TH2, FDOWN, STC, PC, RCH, RR, SNCOVR,
               ESD, SNDENS, SNOWH, SH2O, KDT, FRZFACT, SHDFAC, FLX2, EMISSI, RIBB, lstm_input):
        """
        CALCULATE SOIL MOISTURE AND HEAT FLUX VALUES & UPDATE SOIL MOISTURE
        CONTENT AND SOIL HEAT CONTENT VALUES FOR THE CASE WHEN A SNOW PACK IS
        PRESENT.
        :param ETP:
        :param PRCP:
        :param PRCPF:
        :param SNOWNG:
        :param SMC:
        :param CMC:
        :param DF1:
        :param T1:
        :param SFCTMP:
        :param T24:
        :param TH2:
        :param FDOWN:
        :param STC:
        :param PC:
        :param RCH:
        :param RR:
        :param SNCOVR:
        :param ESD:
        :param SNDENS:
        :param SNOWH:
        :param SH2O:
        :param KDT:
        :param FRZFACT:
        :param SHDFAC:
        :param FLX2:
        :param EMISSI:
        :param RIBB:
        :return:
            STC, SMC, SH2O, DEW, DRIP, ETP, EC, EDIR, ET, ETT, ETNS, ESNOW, FLX1, FLX3, RUNOFF1, RUNOFF2, RUNOFF3,
            SSOIL,
            SNOMLT, CMC, BETA, ESD, SNOWH, SNCOVR, SNDENS, T1
        """
        NSOIL = SH2O.size(0)
        SIGMA = 5.67E-8
        LSUBC = 2.501000E+6
        CPICE = 2.106E+3
        CPH2O = 4.218E+3
        LSUBF = 3.335E+5
        LSUBS = 2.83E+6
        TFREEZ = 273.15
        ESDMIN = 1.E-6
        SNOEXP = 2.0
        EDIR = tensor(0.0)
        EDIR1 = tensor(0.0)
        DEW = tensor(0.0)
        ETT = tensor(0.0)
        ET = torch.zeros(NSOIL)
        ET1 = torch.zeros(NSOIL)
        EC = tensor(0.0)
        EC1 = tensor(0.0)
        PRCP1 = PRCPF * 0.001
        BETA = 1.0
        ETNS = tensor(0.0)
        ESNOW = tensor(0.0)

        # condition = torch.le(ETP, 0)
        # ETP = torch.where(condition & torch.ge(RIBB, 1) & torch.gt(FDOWN, 150.0),
        #                   (torch.min(ETP * (1.0 - RIBB), tensor(0.0)) * SNCOVR / 0.980 + ETP * (0.980 - SNCOVR)) / 0.980,
        #                   ETP)
        # ETP1 = ETP * 0.001
        # DEW = torch.where(condition, -ETP1, tensor(0.0))
        # ETNS1, EDIR, ETT, EC1, ET1 = EVAPO(SMC, CMC, ETP1, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC,
        #                                    CMCMAX,
        #                                    SMCDRY, CFACTR, NROOT, RTDIS, FXEXP)
        # ESNOW2 = torch.where(condition, ETP1 * DT, ETP * SNCOVR * 0.001 * DT)
        # ETANRG = torch.where(condition, ETP * ((1. - SNCOVR) * LSUBC + SNCOVR * LSUBS), ESNOW * LSUBS + ETNS * LSUBC)
        if ETP <= 0:
            if RIBB >= 0.1 and FDOWN > 150.0:
                ETP = (torch.min(ETP * (1.0 - RIBB), tensor(0.0)) * SNCOVR / 0.980 + ETP * (0.980 - SNCOVR)) / 0.980
            if ETP == 0:
                BETA = tensor(0.0)
            ETP1 = ETP * 0.001
            DEW = -ETP1
            ESNOW2 = ETP1 * self.DT
            ETANRG = ETP * ((1. - SNCOVR) * LSUBC + SNCOVR * LSUBS)
        else:
            ETP1 = ETP * 0.001
            if SNCOVR < 1.0:
                ETNS1, EDIR, ETT, EC1, ET1 = self.EVAPO(SMC, CMC, ETP1, SH2O, PC, SHDFAC)
                EDIR1 = EDIR * (1. - SNCOVR)
                EDIR = EDIR1 * 1000
                ET1 = ET1 * (1. - SNCOVR)
                EC1 = EC1 * (1. - SNCOVR)
                EC = EC1 * 1000
                ET = ET1 * 1000
                ETT = ETT * (1. - SNCOVR) * 1000
                ETNS = ETNS1 * (1. - SNCOVR) * 1000

            ESNOW2 = ETP * SNCOVR * 0.001 * self.DT
            ETANRG = ESNOW * LSUBS + ETNS * LSUBC

        FLX1 = torch.where(SNOWNG, CPICE * PRCP * (T1 - SFCTMP),
                           torch.where(torch.gt(PRCP, 0.0), CPH2O * PRCP * (T1 - SFCTMP), 0))

        DSOIL = -(0.5 * self.ZSOIL[0])
        DTOT = SNOWH + DSOIL
        DENOM = 1.0 + DF1 / (DTOT * RR * RCH)
        T12A = ((FDOWN - FLX1 - FLX2 - EMISSI * SIGMA * T24) / RCH + TH2 - SFCTMP - ETANRG / RCH) / RR
        T12B = DF1 * STC[0] / (DTOT * RR * RCH)
        T12 = (SFCTMP + T12A + T12B) / DENOM
        ###################################################
        # condition_1 = torch.le(T12, TFREEZ)
        # condition_2 = torch.le(ESD - ESNOW2, ESDMIN)
        # T1 = torch.where(condition_1, T12, TFREEZ * SNCOVR ** SNOEXP + T12 * (1.0 - SNCOVR ** SNOEXP))
        # SSOIL = torch.where(condition_1, DF1 * (T1 - STC[0]) / DTOT, DF1 * (T1 - STC[0]) / DTOT)
        # ESD = torch.where(condition_1, torch.max(tensor(0.0), ESD - ESNOW2),
        #                   torch.where(condition_2, tensor(0.0), ESD))
        # # SEH = RCH * (T1 - TH2)
        # # T14 = torch.pow(T1, 4)
        # FLX3 = torch.where(condition_1, tensor(0.0), torch.where(condition_2, tensor(0.0), torch.clamp(
        #     FDOWN - FLX1 - FLX2 - EMISSI * SIGMA * torch.pow(T1, 4) - SSOIL - RCH * (T1 - TH2) - ETANRG, min=0.0)))
        # EX = torch.where(condition_1, tensor(0.0), torch.where(condition_2, tensor(0.0), FLX3 * 0.001 / LSUBF))
        # SNOMLT = torch.where(condition_1, tensor(0.0), torch.where(condition_2, tensor(0.0), EX * DT))
        # ESD = torch.where(torch.ge(ESD - SNOMLT, ESDMIN), ESD - SNOMLT, tensor(0.0))
        # EX = torch.where(torch.ge(ESD - SNOMLT, ESDMIN), EX, ESD / DT)
        # FLX3 = torch.where(torch.ge(ESD - SNOMLT, ESDMIN), FLX3, EX * 1000.0 * LSUBF)
        # SNOMLT = torch.where(torch.ge(ESD - SNOMLT, ESDMIN), SNOMLT, ESD)
        # PRCP1 = torch.where(condition_1, PRCP1, PRCP1 + EX)
        if T12 <= TFREEZ:
            T1 = T12
            SSOIL = DF1 * (T1 - STC[0]) / DTOT
            ESD = torch.max(tensor(0.0), ESD - ESNOW2)
            FLX3 = tensor(0.0)
            EX = tensor(0.0)
            SNOMLT = tensor(0.0)
        else:
            T1 = TFREEZ * SNCOVR ** SNOEXP + T12 * (1.0 - SNCOVR ** SNOEXP)
            BETA = tensor(1.0)
            SSOIL = DF1 * (T1 - STC[0]) / DTOT
            if ESD - ESNOW2 <= ESDMIN:
                ESD = tensor(0.0)
                EX = tensor(0.0)
                SNOMLT = tensor(0.0)
                FLX3 = tensor(0.0)
            else:
                ESD = ESD - ESNOW2
                ETP3 = ETP * LSUBC
                SEH = RCH * (T1 - TH2)
                T14 = torch.pow(T1, 4)
                FLX3 = torch.clamp(FDOWN - FLX1 - FLX2 - EMISSI * SIGMA * T14 - SSOIL - SEH - ETANRG, min=0.0)
                EX = FLX3 * 0.001 / LSUBF
                SNOMLT = EX * self.DT
                if ESD - SNOMLT >= ESDMIN:
                    ESD = ESD - SNOMLT
                else:
                    EX = ESD / self.DT
                    FLX3 = EX * 1000.0 * LSUBF
                    SNOMLT = ESD
                    ESD = tensor(0.0)
            PRCP1 = PRCP1 + EX
        SH2O, SMC, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP = self.SMFLX(SMC, CMC, PRCP1, SH2O, KDT, SHDFAC, EDIR1,
                                                                     EC1, ET1)
        ZZ1 = tensor(1.0)
        YY = STC[0] - 0.5 * SSOIL * self.ZSOIL[0] * ZZ1 / DF1
        # T11 = T1
        STC, T1, SSOIL, SH2O = self.SHFLX(STC, SMC, YY, ZZ1, SH2O, DF1)
        if ESD > 0:
            SNOWH, SNDENS = self.SNOWPACK(ESD, SNOWH, SNDENS, T1, YY)
        else:
            ESD = tensor(0.0)
            SNOWH = tensor(0.0)
            SNDENS = tensor(0.0)
            SNCOND = tensor(1.0)
            SNCOVR = tensor(0.0)

        return (
            STC, SMC, SH2O, DEW, DRIP, ETP, EC, EDIR, ET, ETT, ETNS, ESNOW, FLX1, FLX3, RUNOFF1, RUNOFF2, RUNOFF3,
            SSOIL,
            SNOMLT, CMC, BETA, ESD, SNOWH, SNCOVR, SNDENS, T1)


class NoahPy(nn.Module):
    def __init__(self):
        super(NoahPy, self).__init__()
        self.NoahLSMModule = None
        torch.set_default_dtype(torch.float32)
        self.RD = torch.tensor(287.04)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        SFCDIFModule_path = os.path.join(current_dir, "SFCDIFModule.pt")
        NoahLSMModule_path = os.path.join(current_dir, "NoahLSMModule.pt")
        self.SFCDIFModule = torch.jit.load(SFCDIFModule_path)
        self.NoahLSMModule = torch.jit.load(NoahLSMModule_path)
        # self.NoahLSMModule = torch.jit.script(NoahLSMModule())
        # self.SFCDIFModule = torch.jit.script(SFCDIFModule())

    def month_d(self, a12, nowdate):
        """
        Given a set of 12 values, taken to be valid on the fifteenth of each month (Jan through Dec)
        Return a value valid for the day given in <nowdate>, as an interpolation from the 12
        monthly values.
        :param a12:
        :param nowdate:
        :return:
            an interpolation from the 12 monthly values.
        """
        # Convert nowdate to datetime object
        nowy = nowdate.year
        nowm = nowdate.month
        nowd = nowdate.day
        ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Handle leap year by setting the number of days in February for the year in question.
        if nowm == 2:
            if nowy % 4 == 0 and (nowy % 100 != 0 or nowy % 400 == 0):
                ndays[1] = 29  # Leap year

        # Do interpolation between the fifteenth of two successive months.
        prevm, postm, factor = 0, 0, 0
        if nowd == 15:
            return tensor(a12[nowm - 1])
        elif nowd < 15:
            prevm = nowm - 1 if nowm > 1 else 12
            postm = nowm
            factor = (ndays[prevm - 1] - 15 + nowd) / ndays[prevm - 1]
        elif nowd > 15:
            prevm = nowm
            postm = nowm + 1 if nowm < 12 else 1
            factor = (nowd - 15) / ndays[prevm - 1]

        return tensor(a12[prevm - 1] * (1.0 - factor) + a12[postm - 1] * factor)

    def read_forcing_text(self, forcing, k_time):
        TO = 273.15
        CPV = 1870.0
        RV = 461.5
        CW = 4187.0
        ESO = 611.2
        LVH2O = 2.501E6
        eps = 0.622
        specified_row = forcing[k_time]
        # lstm_input_variable = ['windspeed', 'temperature', 'humidity', 'pressure', 'shortwave', 'longwave', 'precipitation',
        #                        'NDVI', 'LAI']
        SFCSPD = specified_row[0]
        WDIR = specified_row[1]
        SFCU = - SFCSPD * torch.sin(WDIR * torch.pi / 180.)
        SFCV = - SFCSPD * torch.cos(WDIR * torch.pi / 180.)
        SFCTMP = specified_row[2]
        RHF = specified_row[3] * 0.01
        SFCPRS = specified_row[4] * 100
        SOLDN = specified_row[5]
        LONGWAVE = specified_row[6]
        PRCP = specified_row[7]
        lstm_input = torch.cat((specified_row[:1], specified_row[2:]), dim=0)
        LW = LVH2O - (CW - CPV) * (SFCTMP - TO)
        svp = ESO * torch.exp(LW * (1.0 / TO - 1.0 / SFCTMP) / RV)
        QS = eps * svp / (SFCPRS - (1. - eps) * svp)
        E = (SFCPRS * svp * RHF) / (SFCPRS - svp * (1. - RHF))
        SPECHUMD = (eps * E) / (SFCPRS - (1.0 - eps) * E)
        SPECHUMD = torch.clamp(SPECHUMD, tensor(0.1E-5), QS * 0.99)
        return SFCSPD, SFCTMP, SPECHUMD, SFCPRS, SOLDN, LONGWAVE, PRCP, SFCU, SFCV, lstm_input

    def CALTMP(self, T1, SFCTMP, SFCPRS, ZLVL, Q2):
        TH2 = SFCTMP + (0.0098 * ZLVL)
        T1V = T1 * (1.0 + 0.61 * Q2)
        TH2V = TH2 * (1.0 + 0.61 * Q2)
        T2V = SFCTMP * (1.0 + 0.61 * Q2)
        RHO = SFCPRS / (self.RD * T2V)
        return TH2, T1V, TH2V, T2V, RHO

    def CALHUM(self, SFCTMP, SFCPRS):
        A2 = torch.tensor(17.67)
        A3 = torch.tensor(273.15)
        A4 = torch.tensor(29.65)
        ELWV = torch.tensor(2.501e6)
        A23M4 = A2 * (A3 - A4)
        E0 = torch.tensor(611.0)
        RV = torch.tensor(461.0)
        EPSILON = torch.tensor(0.622)
        ES = E0 * torch.exp(ELWV / RV * (1. / A3 - 1. / SFCTMP))
        Q2SAT = EPSILON * ES / (SFCPRS - (1 - EPSILON) * ES)
        DQSDT2 = Q2SAT * A23M4 / (SFCTMP - A4) ** 2
        return Q2SAT, DQSDT2




    def open_forcing_file(self, forcing_file_path):
        """
        open forcing file and Initialize the state variable
        :param forcing_file_path:
        :return:

        """
        parameters = {
            "startdate": datetime.now(),
            "enddate": datetime.now(),
            "loop_for_a_while": 0,
            "output_dir": "",
            "Latitude": None,
            "Longitude": None,
            "Forcing_Timestep": 0,
            "Noahlsm_Timestep": 0,
            "Sea_ice_point": False,
            "Soil_layer_thickness": [0.045, 0.046, 0.075, 0.123, 0.204, 0.336, 0.554, 0.913, 0.904, 1, 1, 1, 1, 1, 1, 2,
                                     2,
                                     2],
            "Soil_Temperature": [],
            "Soil_Moisture": [],
            "Soil_Liquid": [],
            "Soil_htype": [],
            "Skin_Temperature": 0,
            "Canopy_water": 0,
            "Snow_depth": 0,
            "Snow_equivalent": 0,
            "Deep_Soil_Temperature": 0,
            "Landuse_dataset": "",
            "Soil_type_index": 3,
            "Vegetation_type_index": 0,
            "Urban_veg_category": 0,
            "glacial_veg_category": 0,
            "Slope_type_index": 0,
            "Max_snow_albedo": 0,
            "Air_temperature_level": 0,
            "Wind_level": 0,
            "Green_Vegetation_Min": 0,
            "Green_Vegetation_Max": 0,
            "Usemonalb": False,
            "Rdlai2d": False,
            "sfcdif_option": 0,
            "iz0tlnd": 0,
            "Albedo_monthly": [0.18, 0.17, 0.16, 0.15, 0.15, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.18],
            "Shdfac_monthly": [0.01, 0.02, 0.07, 0.17, 0.27, 0.58, 0.93, 0.96, 0.65, 0.24, 0.11, 0.02],
            "lai_monthly": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            "Z0brd_monthly": [0.02, 0.02, 0.025, 0.03, 0.035, 0.036, 0.035, 0.03, 0.027, 0.025, 0.02, 0.02]
        }
        with open(forcing_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.__contains__('='):
                    key, value = line.strip().split('=')
                    key = ''.join(key.split())
                    if parameters.__contains__(key):
                        if isinstance(parameters.get(key), list):
                            value = [float(word) for word in value.split()]
                            parameters[key] = value
                        elif isinstance(parameters.get(key), datetime):
                            value = ''.join(value.split())
                            value = value.strip('"')
                            parameters[key] = pd.to_datetime(value, format="%Y%m%d%H%M")
                        elif isinstance(parameters.get(key), str):
                            # value = ''.join(value.split())
                            value = value.strip().strip('"')
                            parameters[key] = value
                        elif isinstance(parameters.get(key), bool):
                            continue
                        else:
                            parameters[key] = float(value.strip())
                elif line.strip().__eq__('/'):
                    break
        output_dir = parameters['output_dir']
        forcing_filename = os.path.basename(forcing_file_path)
        startdate = parameters['startdate']
        enddate = parameters['enddate']
        forcing_timestep = parameters['Forcing_Timestep']
        noahlsm_timestep = int(parameters['Noahlsm_Timestep'])
        ice = parameters['Sea_ice_point']
        T1 = tensor(parameters['Skin_Temperature'])
        STC = tensor(parameters['Soil_Temperature'])
        SMC = tensor(parameters['Soil_Moisture'])
        SH2O = tensor(parameters['Soil_Liquid'], requires_grad=True)
        STYPE = tensor(parameters['Soil_htype'], dtype=torch.int32)
        SLDPTH = tensor(parameters['Soil_layer_thickness'])
        CMC = tensor(parameters['Canopy_water'])
        SNOWH = tensor(parameters['Snow_depth'])
        SNEQV = tensor(parameters['Snow_equivalent'])
        TBOT = tensor(parameters['Deep_Soil_Temperature'])
        VEGTYP = int(parameters['Vegetation_type_index'])
        SLOPETYP = int(parameters['Slope_type_index'])
        SNOALB = tensor(parameters['Max_snow_albedo'])
        ZLVL = tensor(parameters['Air_temperature_level'])
        ZLVL_WIND = tensor(parameters['Wind_level'])
        albedo_monthly = parameters['Albedo_monthly']
        shdfac_monthly = parameters['Shdfac_monthly']
        z0brd_monthly = parameters['Z0brd_monthly']
        lai_monthly = parameters['lai_monthly']
        SHDMIN = tensor(parameters['Green_Vegetation_Min'])
        SHDMAX = tensor(parameters['Green_Vegetation_Max'])
        USEMONALB = tensor(parameters['Usemonalb'])
        RDLAI2D = tensor(parameters['Rdlai2d'])
        IZ0TLND = tensor(parameters['iz0tlnd'])
        sfcdif_option = parameters['sfcdif_option']
        forcing_columns_name = ['Year', 'Month', 'Day', 'Hour', 'minutes', 'windspeed', 'winddir', 'temperature',
                                'humidity', 'pressure', 'shortwave', 'longwave', 'precipitation', 'LAI', 'NDVI']
        x_target_columns = ['windspeed', 'winddir', 'temperature', 'humidity', 'pressure', 'shortwave', 'longwave',
                            'precipitation', 'LAI', 'NDVI']
        forcing_data = pd.read_csv(forcing_file_path, sep='\s+', names=forcing_columns_name, header=None,
                                   skiprows=45)
        forcing_data['Date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day', 'Hour']])
        forcing_data.set_index('Date', inplace=True)
        forcing_data = forcing_data[x_target_columns]
        condition = (forcing_data.index >= startdate) & (forcing_data.index <= enddate)
        forcing_data = forcing_data[condition]
        Date = forcing_data.index
        # print(forcing_file_path)
        # print(forcing_data)
        forcing_data = torch.tensor(forcing_data.to_numpy(), dtype=torch.float32)
        ZSOIL = torch.zeros_like(SH2O)
        NSOIL = SH2O.size(0)
        ZSOIL[0] = -SLDPTH[0]
        for i in range(1, NSOIL):
            ZSOIL[i] = -SLDPTH[i] + ZSOIL[i - 1]

        # NoahLSMModule_ = NoahLSMModule(forcing_filename, VEGTYP, ZSOIL, SLDPTH, SHDMIN, SHDMAX, STYPE, noahlsm_timestep, TBOT, SLOPETYP)
        # self.NoahLSMModule = torch.jit.script(NoahLSMModule_)
        # self.NoahLSMModule = NoahLSMModule_
        self.NoahLSMModule.set_parameter(VEGTYP, ZSOIL, SLDPTH, SHDMIN, SHDMAX, STYPE, noahlsm_timestep, TBOT, SLOPETYP)
        # torch.jit.save(self.NoahLSMModule, "NoahLSMModule.pt")

        return (Date, forcing_data, output_dir, forcing_filename, startdate, enddate, noahlsm_timestep,
                ice, T1, STC, SMC, SH2O,
                CMC, SNOWH, SNEQV, TBOT, SNOALB, ZLVL, ZLVL_WIND,
                albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly
                , USEMONALB, RDLAI2D, IZ0TLND, sfcdif_option)

    #################################
    # FXEXP_Data, FRZK_Data, CZIL_data, EMISSMIN, EMISSMAX, ALBEDOMIN,ALBEDOMAX, BB,MAXSMC, QTZ,
    ##############################

    def noah_main(self, file_name, trained_parameter=None, lstm_model=None, output_flag=False):
        torch.set_default_dtype(torch.float32)
        badval = -1.E36
        PC = tensor(badval)  # Plant coefficient, where PC * ETP = ETA ( Fraction [0.0-1.0] )
        RCS = tensor(badval)  # Incoming solar RC factor ( dimensionless )
        RCT = tensor(badval)  # Air temperature RC factor ( dimensionless )
        RCQ = tensor(badval)  # Atmospheric water vapor deficit RC factor ( dimensionless )
        RCSOIL = tensor(badval)  # Soil moisture RC factor ( dimensionless )
        Q1 = tensor(badval)  # Effective mixing ratio at the surface ( kg kg{-1} )
        SNOTIME1 = tensor(0.0)  # Age of the snow on the ground.


        """read forcing file"""
        (Date, forcing_data, output_dir, forcing_filename, startdate, enddate, noahlsm_timestep,
         ice, T1, STC, SMC, SH2O,
         CMC, SNOWH, SNEQV, TBOT, SNOALB, ZLVL, ZLVL_WIND,
         albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly
         , USEMONALB, RDLAI2D, IZ0TLND, sfcdif_option) = self.open_forcing_file(file_name)

        if trained_parameter is not None:
            self.NoahLSMModule.set_grad_soil_param(trained_parameter)

        # CMC = tensor(0.0)
        # SNOWH = tensor(0.0)
        # SNEQV = tensor(0.0)
        # SNOALB = tensor(0.75)
        # ZLVL = tensor(2.0)
        # ZLVL_WIND = tensor(10.0)
        # albedo_monthly = tensor([0.18, 0.17, 0.16, 0.15, 0.15, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.18])
        # shdfac_monthly = tensor([[0.01, 0.02, 0.07, 0.17, 0.27, 0.58, 0.93, 0.96, 0.65, 0.24, 0.11, 0.02]])
        # lai_monthly = tensor([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        # z0brd_monthly = tensor([0.02, 0.02, 0.025, 0.03, 0.035, 0.036, 0.035, 0.03, 0.027, 0.025, 0.02, 0.02])
        # IZ0TLND = tensor(0)
        # USEMONALB = tensor(False)
        # RDLAI2D = tensor(False)
        timestep = pd.Timedelta(seconds=noahlsm_timestep)
        EMISSI = 0.96
        ALBEDO = self.month_d(albedo_monthly, startdate)
        Z0 = self.month_d(z0brd_monthly, startdate)
        if sfcdif_option == 1:
            self.SFCDIFModule.MYJSFCINIT()
            Z0BRD = Z0
        else:
            Z0BRD = torch.tensor(badval)
        # CZIL = self.gen_parameters['CZIL_DATA']
        CZIL = tensor(0.1)
        CH = tensor(1.E-4)
        CM = tensor(1.E-4)
        nowdate = startdate
        k_time = 0
        out_SMC = []
        out_STC = []
        out_SH2O = []
        while nowdate <= enddate:
            SFCSPD, SFCTMP, Q2, SFCPRS, SOLDN, LONGWAVE, PRCP, SFCU, SFCV, lstm_input = self.read_forcing_text(
                forcing_data, k_time)
            FFROZP = torch.where(torch.gt(PRCP, 0) & torch.lt(SFCTMP, 273.15), torch.tensor(1), torch.tensor(0))
            TH2, T1V, TH2V, T2V, RHO = self.CALTMP(T1, SFCTMP, SFCPRS, ZLVL, Q2)
            Q2SAT, DQSDT2 = self.CALHUM(SFCTMP, SFCPRS)  # Returns Q2SAT, DQSDT2 PENMAN needed
            ALB = torch.where(USEMONALB, self.month_d(albedo_monthly, nowdate), torch.tensor(badval))
            XLAI = torch.where(RDLAI2D, self.month_d(lai_monthly, nowdate), torch.tensor(badval))
            SHDFAC = self.month_d(shdfac_monthly, nowdate)
            Q1 = torch.where(Q1 == badval, Q2, Q1)
            if sfcdif_option == 1:
                RIBB, CM, CH = self.SFCDIFModule.SFCDIF_MYJ(ZLVL, ZLVL_WIND, Z0, Z0BRD, SFCPRS, T1, SFCTMP, Q1, Q2,
                                                            SFCSPD, CZIL, CM,
                                                            CH, IZ0TLND)
            else:
                RIBB, CH = self.SFCDIFModule.SFCDIF_MYJ_Y08(Z0, ZLVL_WIND, ZLVL, SFCSPD, T1, SFCTMP, Q2, SFCPRS)

            SOLNET = SOLDN * (1.0 - ALBEDO)
            LWDN = LONGWAVE * EMISSI
            (CMC, T1, STC, SMC, SH2O, SNOWH, SNEQV, SNOTIME1, ALBEDO,
             PC, Q1, Z0, Z0BRD, EMISSI) = self.NoahLSMModule.SFLX(FFROZP, LWDN, SOLDN, SOLNET, SFCPRS, PRCP, SFCTMP, Q2,
                                                                  TH2, Q2SAT, DQSDT2, SHDFAC,
                                                                  ALB, SNOALB, CMC, T1, STC, SMC, SH2O, SNOWH, SNEQV,
                                                                  CH, PC, XLAI, SNOTIME1, RIBB,
                                                                  k_time, lstm_input)

            """output"""
            out_STC.append(STC - 273.15)
            out_SH2O.append(SH2O)

            # CMC, T1, STC, SMC, SH2O,
            # SNOWH, SNEQV,SNOTIME1,
            # PC, RCS, RCT, RCQ, RCSOIL,
            # Q1, Z0, Z0BRD, EMISSI
            k_time += 1
            nowdate = nowdate + timestep
            if k_time % 30 == 0:
                SH2O = SH2O.detach()
                STC = STC.detach()

        """output"""
        NSOIL = SH2O.size(0)
        if output_flag:
            SH2O_columns = [f'SH2O({i + 1})' for i in range(NSOIL)]
            STC_columns = [f'STC({i + 1})' for i in range(NSOIL)]
            SMC_columns = [f'SMC({i + 1})' for i in range(NSOIL)]
            out_columns = (STC_columns +
                           SH2O_columns +
                           SMC_columns)
            out = torch.cat([torch.stack(out_STC),
                             torch.stack(out_SH2O),
                             torch.stack(out_SMC),
                             ], dim=1)
            pd.DataFrame(out.detach().numpy(), columns=out_columns,
                         index=Date[:k_time]).to_csv(os.path.join(output_dir, "NoahPy_output.txt"), index=True, sep=' ')
        condition = (Date >= startdate) & (Date <= enddate)
        return Date[condition], torch.stack(out_STC), torch.stack(out_SH2O)




