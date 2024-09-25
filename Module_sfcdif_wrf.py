'''
@Project ：NoahPy
@File    ：Module_sfcdif_wrf.py
@Author  ：tianwb
@Date    ：2024/5/14 18:08
'''

from torch import tensor, Tensor
from Module_model_constants import *

torch.set_default_dtype(torch.float32)
ITRMX = torch.tensor(5)

EXCML = torch.tensor(0.0001)
EXCMS = torch.tensor(0.0001)
VKARMAN = torch.tensor(0.4)
ZTFC = torch.tensor(1.0)
ELOCP = torch.tensor(2.72e6) / CP
EPSU2 = torch.tensor(1.0e-6)
EPSUST = torch.tensor(1.0e-9)
SQVISC = torch.tensor(258.2)
RIC = torch.tensor(0.505)
EPSZT = torch.tensor(1.0e-28)
RD = tensor(287.0)
KZTM = torch.tensor(10001)
KZTM2 = KZTM - 2

WWST = torch.tensor(1.2)
WWST2 = WWST * WWST

PSIM2 = torch.zeros(KZTM)
PSIH2 = torch.zeros(KZTM)
ZTMAX2 = torch.zeros(0)
DZETA2 = torch.zeros(0)
ZTMIN2 = torch.tensor(-5.0)


def MYJSFCINIT():
    global ZTMIN2, ZTMAX2, DZETA2, PSIH2, PSIM2
    # Parameter definitions
    PIHF = torch.pi / 2.
    EPS = tensor(1.0e-6)
    ZTMIN1 = tensor(-5.0)

    # Variable initialization
    ZTMAX1 = tensor(1.0)
    ZTMAX2 = tensor(1.0)
    ZRNG1 = ZTMAX1 - ZTMIN1
    ZRNG2 = ZTMAX2 - ZTMIN2
    DZETA1 = ZRNG1 / (KZTM - 1)
    DZETA2 = ZRNG2 / (KZTM - 1)
    ZETA1 = ZTMIN1.clone()
    ZETA2 = ZTMIN2.clone()

    # Function definition loop
    for K in range(1, KZTM + 1):
        if ZETA2 < 0.:
            X = torch.sqrt(torch.sqrt(1. - 16. * ZETA2))
            PSIM2[K - 1] = -2. * torch.log((X + 1.) / 2.) - torch.log((X * X + 1.) / 2.) + 2. * torch.arctan(X) - PIHF
            PSIH2[K - 1] = -2. * torch.log((X * X + 1.) / 2.)
        else:
            PSIM2[K - 1] = 0.7 * ZETA2 + 0.75 * ZETA2 * (6. - 0.35 * ZETA2) * torch.exp(-0.35 * ZETA2)
            PSIH2[K - 1] = 0.7 * ZETA2 + 0.75 * ZETA2 * (6. - 0.35 * ZETA2) * torch.exp(-0.35 * ZETA2)

        if K == KZTM:
            ZTMAX1 = ZETA1
            ZTMAX2 = ZETA2

        ZETA1 = ZETA1 + DZETA1
        ZETA2 = ZETA2 + DZETA2

    ZTMAX1 -= EPS
    ZTMAX2 -= EPS


def SFCDIF_MYJ(ZSL, ZSL_WIND, Z0, Z0BASE, SFCPRS, TZ0, TLOW, QZ0, QLOW, SFCSPD, CZIL, AKMS, AKHS, IZ0TLND):
    THLOW = TLOW * (P0 / SFCPRS) ** RCP
    THZ0 = TZ0 * (P0 / SFCPRS) ** RCP
    THELOW = THLOW
    CXCHL = EXCML / ZSL
    BTGX = G / THLOW
    ELFC = VKARMAN * BTGX
    BTGH = BTGX * 1000.
    THM = (THELOW + THZ0) * 0.5
    TEM = (TLOW + TZ0) * 0.5
    A = THM * P608
    B = (ELOCP / TEM - 1. - P608) * THM
    CWMLOW = torch.tensor(0.0)
    DTHV = ((THELOW - THZ0) * ((QLOW + QZ0 + CWMLOW) * (0.5 * P608) + 1.) + (QLOW - QZ0 + CWMLOW) * A + CWMLOW * B)
    DU2 = torch.max(SFCSPD * SFCSPD, EPSU2)
    RIB = BTGX * DTHV * ZSL_WIND * ZSL_WIND / DU2 / ZSL
    ZU = Z0
    ZT = ZU * ZTFC
    ZSLU = ZSL_WIND + ZU
    RZSU = Tensor(ZSLU / ZU)
    RLOGU = torch.log(RZSU)
    ZSLT = ZSL + ZU
    CZIL_LOCAL = 10.0 ** (-0.40 * (Z0 / 0.07))
    ZILFC = torch.where(torch.eq(IZ0TLND, 0), -CZIL * VKARMAN * SQVISC, -CZIL_LOCAL * VKARMAN * SQVISC)
    CZETMAX = 10.
    ZZIL = torch.where(torch.gt(DTHV, 0),
                       torch.where(torch.lt(RIB, RIC), ZILFC * (1.0 + (RIB / RIC) * (RIB / RIC) * CZETMAX),
                                   ZILFC * (1.0 + CZETMAX)), ZILFC)
    WSTAR2 = torch.where(BTGH * AKHS * DTHV != 0.0, WWST2 * torch.abs(BTGH * AKHS * DTHV) ** (2.0 / 3.0), tensor(0.0))
    # if DTHV > 0.0:
    #     if RIB < RIC:
    #         ZZIL = ZILFC * (1.0 + (RIB / RIC) * (RIB / RIC) * CZETMAX)
    #     else:
    #         ZZIL = ZILFC * (1.0 + CZETMAX)
    # else:
    #     ZZIL = ZILFC

    # if BTGH * AKHS * DTHV != 0.0:
    #     WSTAR2 = WWST2 * torch.abs(BTGH * AKHS * DTHV) ** (2.0 / 3.0)
    # else:
    #     WSTAR2 = 0.0

    USTAR = torch.max(torch.sqrt(AKMS * torch.sqrt(DU2 + WSTAR2)), EPSUST)
    ITRMX = 5
    for ITR in range(0, ITRMX):
        ZT = torch.max(torch.exp(ZZIL * torch.sqrt(USTAR * Z0BASE)) * Z0BASE, EPSZT)
        RZST = ZSLT / ZT
        RLOGT = torch.log(RZST)
        RLMO = ELFC * AKHS * DTHV / USTAR ** 3
        ZETALU = ZSLU * RLMO
        ZETALT = ZSLT * RLMO
        ZETAU = ZU * RLMO
        ZETAT = ZT * RLMO

        ZETALU = torch.min(torch.max(ZETALU, ZTMIN2), ZTMAX2)
        ZETALT = torch.min(torch.max(ZETALT, ZTMIN2), ZTMAX2)
        ZETAU = torch.min(torch.max(ZETAU, ZTMIN2 / RZSU), ZTMAX2 / RZSU)
        ZETAT = torch.min(torch.max(ZETAT, ZTMIN2 / RZST), ZTMAX2 / RZST)
        #
        RZ = (ZETAU - ZTMIN2) / DZETA2
        K = torch.floor(RZ).int()
        RDZT = RZ - K.float()
        K = torch.min(K, KZTM2).max(torch.tensor(0))
        PSMZ = (PSIM2[K + 1] - PSIM2[K]) * RDZT + PSIM2[K]
        #
        RZ = (ZETALU - ZTMIN2) / DZETA2
        K = torch.floor(RZ).int()
        RDZT = RZ - K.float()
        K = torch.min(K, KZTM2).max(torch.tensor(0))
        PSMZL = (PSIM2[K + 1] - PSIM2[K]) * RDZT + PSIM2[K]
        #
        SIMM = PSMZL - PSMZ + RLOGU
        #
        RZ = (ZETAT - ZTMIN2) / DZETA2
        K = torch.floor(RZ).int()
        RDZT = RZ - K.float()
        K = torch.min(K, KZTM2).max(torch.tensor(0))
        PSHZ = (PSIH2[K + 1] - PSIH2[K]) * RDZT + PSIH2[K]
        #
        RZ = (ZETALT - ZTMIN2) / DZETA2
        K = torch.floor(RZ).int()
        RDZT = RZ - K.float()
        K = torch.min(K, KZTM2).max(torch.tensor(0))
        PSHZL = (PSIH2[K + 1] - PSIH2[K]) * RDZT + PSIH2[K]
        #
        SIMH = PSHZL - PSHZ + RLOGT
        USTARK = USTAR * VKARMAN
        AKMS = torch.max(USTARK / SIMM, CXCHL)
        AKHS = torch.max(USTARK / SIMH, CXCHL)

        WSTAR2 = torch.where(
            DTHV <= 0.0,
            WWST2 * torch.abs(BTGH * AKHS * DTHV) ** (2.0 / 3.0),
            torch.tensor(0.0)
        )
        USTAR = torch.max(torch.sqrt(AKMS * torch.sqrt(DU2 + WSTAR2)), EPSUST)
    return RIB, AKMS, AKHS


def SFCDIF_MYJ_Y08(z0m, zm, zh, wspd1, tsfc, tair, qair, psfc):
    # Parameters
    excm = tensor(0.001)
    aa = tensor(0.007)
    p0 = tensor(1.0e5)

    # Local variables
    wspd = torch.max(wspd1, tensor(0.01))
    rhoair = psfc / (RD * tair * (1 + 0.61 * qair))
    ptair = tair * (psfc / (psfc - rhoair * 9.81 * zh)) ** RCP
    ptsfc = tsfc

    pt1 = ptair

    c_u = 0.4 / torch.log(zm / z0m)  # von Karman constant / log(zm / z0m)
    c_pt = 0.4 / torch.log(zh / z0m)  # von Karman constant / log(zh / z0m)

    tstr = c_pt * (pt1 - ptsfc)  # tstr for stable case
    ustr = c_u * wspd  # ustr for stable case

    lmo = ptair * ustr ** 2 / (0.4 * 9.81 * tstr)  # Monin Obukhov length

    nu = 1.328e-5 * (p0 / psfc) * (pt1 / 273.15) ** 1.754  # viscosity of air

    for i in range(3):
        z0h = z0mz0h(zh, z0m, nu, ustr, tstr)
        c_u, c_pt, ribbb = flxpar(zm, zh, z0m, z0h, wspd, ptsfc, pt1)

        ustr = c_u * wspd
        tstr = c_pt * (pt1 - ptsfc)

    c_pt = torch.where(Tensor(torch.abs(ptair - ptsfc) < 0.001), c_pt, tstr / (ptair - ptsfc))

    ra = 1 / (ustr * c_pt)

    chh = 1.0 / ra

    chh = torch.max(chh, excm * (1.0 / zm))
    # chh = max(chh, aa)  # Uncomment if needed

    return ribbb, chh


def z0mz0h(zh, z0m, nu, ustr, tstr):
    # Constants
    a = tensor(70.0)

    # Parameters
    b = tensor(-7.2)

    # Calculate z0h
    z0h = a * nu / ustr * torch.exp(b * torch.sqrt(ustr) * torch.sqrt(torch.sqrt(torch.abs(-tstr))))
    z0h = torch.min(zh / 10, torch.max(z0h, tensor(1.0E-10)))

    return z0h


def flxpar(zm, zh, z0m, z0h, wspd, ptsfc, pt1):
    lmo, ribb1 = MOlength(zm, zh, z0m, z0h, wspd, ptsfc, pt1)

    c_u, c_pt = CuCpt(lmo, z0m, z0h, zm, zh)

    return c_u, c_pt, ribb1


def MOlength(zm, zh, z0m, z0h, wspd, ptsfc, pt1):
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
        bulkri = min(bulkri, 0.2,
                     prantl01 * betah * (1 - z0h / zh) / betam ** 2 / (1 - z0m / zm) - 0.05)
        d = bulkri / prantl01
        a = d * betam ** 2 * (zm - z0m) - betah * (zh - z0h)
        b = 2 * d * betam * torch.log(zm / z0m) - torch.log(zh / z0h)
        c = d * torch.log(zm / z0m) ** 2 / (zm - z0m)
        lmo = (-b - torch.sqrt(torch.abs(b ** 2 - 4 * a * c))) / (2 * a)

    if 0 < lmo < 1.0e-6:
        lmo = 1.0e-6
    elif 0 > lmo > -1.0e-6:
        lmo = -1.0e-6

    lmo = 1 / lmo

    return lmo, bulkri


def CuCpt(lmo, zm1, zh1, zm2, zh2):
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
