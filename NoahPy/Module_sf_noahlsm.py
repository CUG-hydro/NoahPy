'''
@Project ：NoahPy
@File    ：Module_sf_noahlsm.py
@Author  ：tianwb
@Date    ：2024/5/14 18:08
'''
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import tensor, Tensor, nn

from Module_model_constants import EMISSI_S, CP, BARE, TFREEZ


# from distributed_parallel_train import LSTM


class SoilParam(nn.Module):
    def __init__(self):
        super(SoilParam, self).__init__()
        target_columns = ['BB', 'MAXSMC', 'SATDK', 'SATPSI', 'QTZ']
        soil_parameters = soil_parameter[target_columns]
        self.BB = nn.Parameter(tensor(soil_parameters['BB'].to_numpy(), dtype=torch.float32))
        self.MAXSMC = nn.Parameter(tensor(soil_parameters['MAXSMC'].to_numpy(), dtype=torch.float32))
        self.SATDK = nn.Parameter(tensor(soil_parameters['SATDK'].to_numpy(), dtype=torch.float32))
        self.SATPSI = nn.Parameter(tensor(soil_parameters['SATPSI'].to_numpy(), dtype=torch.float32))
        self.QTZ = nn.Parameter(tensor(soil_parameters['QTZ'].to_numpy(), dtype=torch.float32))

    def get_by_index(self, indices, required_grad=True):
        index = indices - 1
        bb = self.BB[index].squeeze()
        maxSMC = self.MAXSMC[index].squeeze()
        SATDK = self.SATDK[index].squeeze()
        SATPSI = self.SATPSI[index].squeeze()
        return bb, maxSMC, SATDK, SATPSI,

    def get_BB_by_index(self, indices, required_grad=True):
        index = indices - 1
        bb = self.BB[index].squeeze()
        return bb

    def get_MAXSMC_by_index(self, indices, required_grad=True):
        index = indices - 1
        maxSMC = self.MAXSMC[index].squeeze()
        SATDK = self.SATDK[index].squeeze()
        return maxSMC

    def get_SATDK_by_index(self, indices, required_grad=True):
        index = indices - 1
        SATDK = self.SATDK[index].squeeze()
        return SATDK

    def get_SATPSI_by_index(self, indices, required_grad=True):
        index = indices - 1
        SATPSI = self.SATPSI[index].squeeze()
        return SATPSI

    def get_QTZ_by_index(self, indices, required_grad=True) -> Tensor:
        index = indices - 1
        QTZ = self.QTZ[index].squeeze()
        return QTZ

    def get_all_by_index(self, indices, required_grad=True):
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


veg_param_path = "parameter_new/VEGPARM.TBL"
soil_param_path = "parameter_new/SOILPARM.TBL"
veg_parameter = pd.read_csv(veg_param_path, sep=r',\s*', engine='python', header=0, index_col=0, usecols=range(16),
                            dtype=np.float32)
soil_parameter = pd.read_csv(soil_param_path, sep=r',\s*', engine='python', header=0, index_col=0, usecols=range(11),
                             dtype=np.float32)
grad_soil_parameter: Optional[tuple] = None
NSOIL: Optional[int] = 20
NROOT: Optional[int] = 2
lstm_model: None
hc: Optional[tensor] = None
Soil_Parameter = SoilParam()
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


@torch.jit.ignore
def REDSTP(STPNUM: Tensor, target: Tensor, required_grad: bool = True) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    read soil parameters
    :param target:
    :param required_grad:
    :param STPNUM:  soil type
    :return:
        PSISAT, BEXP, DKSAT, DWSAT,
        SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ
    """
    train_layer_size = 10
    if grad_soil_parameter is not None:
        QUARTZ = Soil_Parameter.get_QTZ_by_index(STPNUM)
        if required_grad:
            if len(target) > 10:
                # BB, MAXSMC, SATPSI, SATDK
                first_ten_layer_BEXP, first_ten_layer_SMCMAX, first_ten_layer_DKSAT, first_ten_layer_PSISAT = grad_soil_parameter
                other_layer_BEXP, other_layer_SMCMAX, other_layer_DKSAT, other_layer_PSISAT = Soil_Parameter.get_by_index(
                    STPNUM[train_layer_size:])
                BEXP = torch.concat([first_ten_layer_BEXP, other_layer_BEXP])
                SMCMAX = torch.concat([first_ten_layer_SMCMAX, other_layer_SMCMAX])
                DKSAT = torch.concat([first_ten_layer_DKSAT, other_layer_DKSAT])
                PSISAT = torch.concat([first_ten_layer_PSISAT, other_layer_PSISAT])
            else:
                index = torch.tensor(STPNUM) - 1
                BEXP, SMCMAX, DKSAT, PSISAT = [item[index] for item in grad_soil_parameter]
        else:
            BEXP, SMCMAX, DKSAT, PSISAT = Soil_Parameter.get_by_index(STPNUM)
    else:
        BEXP, SMCMAX, DKSAT, PSISAT, QUARTZ = Soil_Parameter.get_all_by_index(STPNUM, required_grad)

    F1 = torch.log10(PSISAT) + BEXP * torch.log10(SMCMAX) + 2.0
    if __debug__:
        assert not torch.isnan(F1).any(), "Input tensor contains NaN"

    REFSMC1 = SMCMAX * (5.79E-9 / DKSAT) ** (1 / (2 * BEXP + 3))
    SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
    if __debug__:
        assert not torch.isnan(SMCREF).any(), "Input tensor contains NaN"
    WLTSMC1 = SMCMAX * (200. / PSISAT) ** (-1. / BEXP)

    SMCWLT = 0.5 * WLTSMC1
    if __debug__:
        assert not torch.isnan(SMCWLT).any(), "Input tensor contains NaN"
    DWSAT = BEXP * DKSAT * PSISAT / SMCMAX
    if __debug__:
        assert not torch.isnan(DWSAT).any(), "Input tensor contains NaN"
    SMCDRY = SMCWLT
    return PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ


# options for supercooled liquid water (or ice fraction)
# 1 -> Noah default, Koren's iteration
# 2 -> no iteration (Niu and Yang, 2006 JHM);
OPT_FRZ = 1

# options for frozen soil permeability
# 1 -> Noah default, nonlinear effects, less permeable (old)  //no ice
# 2 -> New parametric scheme,linear effects, more permeable (Niu and Yang, 2006, JHM)  //have ice
OPT_INF = 2  # (suggested 2 ice)

# options for soil thermal conductivity
# 1 -> Noah default,   // no gravel
# 2 -> New parametric scheme,   //have gravel & bedrock
OPT_TCND = 2  # (suggested 2 gravel)

now_date = None
k_time = None
grads = {}


def register_hooks(var, name):
    """
    Register a hook function to save the gradient of the variable during backpropagation
    :param var:
    :param name:
    :return:
    """

    def hook(grad):
        if name in grads:
            grads[name] += grad
        else:
            grads[name] = grad

    var.register_hook(hook)


def SFLX(FFROZP, DT, SLDPTH, ZSOIL
         , LWDN, SOLDN, SOLNET, SFCPRS, PRCP, SFCTMP, Q2,
         TH2, Q2SAT, DQSDT2, VEGTYP: int, SLOPETYP: int, SHDFAC, SHDMIN, SHDMAX,
         ALB, SNOALB, TBOT, CMC, T1, STC, SMC, SH2O, STYPE, SNOWH, SNEQV, CH
         , PC, XLAI, RDLAI2D, USEMONALB, SNOTIME1, RIBB, nowdate, ktime: int,
         lstm_input):
    """
    NOAH Physical Process Driver
    :param FFROZP:
    :param DT:
    :param SLDPTH:
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
    :param VEGTYP:
    :param SLOPETYP:
    :param SHDFAC:
    :param SHDMIN:
    :param SHDMAX:
    :param ALB:
    :param SNOALB:
    :param TBOT:
    :param CMC:
    :param T1:
    :param STC:
    :param SMC:
    :param SH2O:
    :param STYPE:
    :param SNOWH:
    :param SNEQV:
    :param CH:
    :param PC:
    :param XLAI:
    :param RDLAI2D:
    :param USEMONALB:
    :param SNOTIME1:
    :param RIBB:
    :return:
    """
    """global parameter"""
    global now_date, k_time, grad_soil_parameter
    now_date = nowdate
    k_time = ktime

    with torch.no_grad():
        # Read parameters
        (CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA, SHDFAC, RSMIN, RGL, HS, ZBOT,
         FRZX, PSISAT, SLOPE, SNUP, SALP, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, FXEXP,
         RTDIS, NROOT, CZIL, LAIMIN, LAIMAX, EMISSMIN, EMISSMAX, ALBEDOMIN, ALBEDOMAX, Z0MIN,
         Z0MAX, CSOIL, PTU, LVCOEF) = REDPRM(VEGTYP, STYPE[0], SLOPETYP, SLDPTH, ZSOIL, SHDFAC)

        PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE[0], torch.arange(1))

        condition_1 = torch.ge(SHDFAC, SHDMAX)
        condition_2 = torch.le(SHDFAC, SHDMIN)
        INTERP_FRACTION = torch.clip((SHDFAC - SHDMIN) / (SHDMAX - SHDMIN), 0.0, 1.0)
        EMBRD = torch.where(condition_1, EMISSMAX, torch.where(condition_2, EMISSMIN, (
                1.0 - INTERP_FRACTION) * EMISSMIN + INTERP_FRACTION * EMISSMAX))
        XLAI = torch.where(condition_1, LAIMAX,
                           torch.where(condition_2, LAIMIN, torch.where(torch.logical_not(RDLAI2D), (
                                   1.0 - INTERP_FRACTION) * LAIMIN + INTERP_FRACTION * LAIMAX, XLAI)))
        ALB = torch.where(condition_1, ALBEDOMIN,
                          torch.where(condition_2, ALBEDOMAX,
                                      torch.where(torch.logical_not(USEMONALB), (
                                              1.0 - INTERP_FRACTION) * ALBEDOMAX + INTERP_FRACTION * ALBEDOMIN, ALB)))
        Z0BRD = torch.where(condition_1, Z0MAX,
                            torch.where(condition_2, Z0MIN,
                                        (1.0 - INTERP_FRACTION) * Z0MIN + INTERP_FRACTION * Z0MAX))

        # ----------------------------------------------------------------------
        # Initialize the precipitation logical variable
        # ----------------------------------------------------------------------

        if SNEQV <= 1.0e-7:
            SNEQV = torch.tensor(0.0)  # snow water equivalent
            SNDENS = torch.tensor(0.0)
            SNOWH = torch.tensor(0.0)  # snow depth
            SNCOND = torch.tensor(1.0)
        else:
            SNDENS = SNEQV / SNOWH  # snow density
            if SNDENS > 1.0:
                raise ValueError('Physical snow depth is less than snow water equiv.')
            SNCOND = CSNOW(SNDENS)

        SNOWNG = torch.where(torch.logical_and(torch.gt(PRCP, 0.0), torch.gt(FFROZP, 0.5)), tensor(True),
                             torch.tensor(False))
        FRZGRA = torch.where(torch.logical_and(torch.gt(PRCP, 0.0), torch.le(T1, TFREEZ)), tensor(True),
                             torch.tensor(False))

        if SNOWNG or FRZGRA:
            SN_NEW = PRCP * DT * 0.001  # from kg/m^2*s convert to m
            SNEQV = SNEQV + SN_NEW  # snow water equivalent
            PRCPF = tensor(0.0)
            SNDENS, SNOWH = SNOW_NEW(SFCTMP, SN_NEW, SNOWH, SNDENS)
            SNCOND = CSNOW(SNDENS)
        else:
            PRCPF = PRCP

        DSOIL = -(0.5 * ZSOIL[0])
        DF1 = TDFCND_C05_Tensor(SMC[0].detach(), QUARTZ, SMCMAX, SH2O[0].detach(), STYPE[0])
        if SNEQV == 0.0:
            SNCOVR = tensor(0.0)
            ALBEDO = ALB
            EMISSI = EMBRD
            SSOIL = DF1 * (T1 - STC[0]) / DSOIL
        else:
            SNCOVR = SNFRAC(SNEQV, SNUP, SALP)
            SNCOVR = torch.clamp(SNCOVR, min=0.0, max=1.0)
            ALBEDO, EMISSI, SNOTIME1 = ALCALC(ALB, SNOALB, EMBRD, SNCOVR, DT, SNOWNG, SNOTIME1, LVCOEF)
            DF1 = torch.where(torch.gt(SNCOVR, 0.97), SNCOND, DF1 * torch.exp(SBETA * SHDFAC))
            DTOT = SNOWH + DSOIL
            DF1A = SNOWH / DTOT * SNCOND + DSOIL / DTOT * DF1
            DF1 = DF1A * SNCOVR + DF1 * (1.0 - SNCOVR)
            SSOIL = DF1 * (T1 - STC[0]) / DTOT

        Z0 = torch.where(torch.gt(SNCOVR, 0.), SNOWZ0(SNCOVR, Z0BRD, SNOWH), Z0BRD)

        FDOWN = SOLNET + LWDN
        T2V = SFCTMP * (1.0 + 0.61 * Q2)
        # ETP: Potential Evapotranspiration
        EPSCA, ETP, RCH, RR, FLX2, T24 = PENMAN(SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT, SNOWNG,
                                                FRZGRA,
                                                DQSDT2,
                                                EMISSI, SNCOVR)

        if SHDFAC > 0.:
            # canopy resistance and plant coefficient
            PC, RC, RCS, RCT, RCQ, RCSOIL = CANRES(SOLDN, CH, SFCTMP, Q2, SFCPRS, SH2O, ZSOIL,
                                                   STYPE, RSMIN, Q2SAT, DQSDT2,
                                                   TOPT, RSMAX, RGL, HS, XLAI, EMISSI, NROOT)

    if SNEQV == 0.0:
        STC, SMC, SH2O, DEW, DRIP, EC, EDIR, ETA, ET, ETT, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1, FLX1, FLX3 = NOPAC(
            STYPE, ETP, PRCP, SMC, SMCMAX, SMCDRY, CMC, CMCMAX, DT, SHDFAC, SBETA,
            SFCTMP, T24, TH2, FDOWN, EMISSI, STC, EPSCA, PC, RCH, RR, CFACTR, SH2O, SLOPE, KDT, FRZX, ZSOIL, TBOT,
            ZBOT, NROOT, RTDIS, QUARTZ, FXEXP, CSOIL, lstm_input)
        ETA_KINEMATIC = ETA
    else:
        (STC, SMC, SH2O, DEW, DRIP, ETP, EC, EDIR, ET, ETT, ETNS, ESNOW, FLX1, FLX3, RUNOFF1, RUNOFF2, RUNOFF3,
         SSOIL, SNOMLT, CMC, BETA, SNEQV, SNOWH, SNCOVR, SNDENS, T1) = SNOPAC(STYPE, ETP, PRCP, PRCPF, SNOWNG, SMC,
                                                                              SMCMAX, SMCDRY, CMC,
                                                                              CMCMAX, DT, DF1, T1, SFCTMP, T24,
                                                                              TH2, FDOWN, STC, PC, RCH, RR,
                                                                              CFACTR, SNCOVR, SNEQV, SNDENS, SNOWH,
                                                                              SH2O, SLOPE, KDT, FRZX, ZSOIL, TBOT, ZBOT,
                                                                              SHDFAC, NROOT, RTDIS, FXEXP,
                                                                              CSOIL, FLX2, EMISSI, RIBB, lstm_input)
        ETA_KINEMATIC = ESNOW + ETNS
    # ----------------------------------------------------------------------
    #   PREPARE SENSIBLE HEAT (H) FOR RETURN TO PARENT MODEL
    # ----------------------------------------------------------------------
    Q1 = Q2 + ETA_KINEMATIC * CP / RCH
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

def DEVAP(ETP1, SMC, SHDFAC, SMCMAX, SMCDRY, FXEXP) -> Tensor:
    """
    CALCULATE DIRECT SOIL EVAPORATION
    :param ETP1: Potential evapotranspiration
    :param SMC: SOIL MOISTURE
    :param SHDFAC: Areal fractional coverage of green vegetation
    :param SMCMAX: saturated value of soil moisture
    :param SMCDRY: Dry soil moisture threshold where direct evaporation from the top layer ends
    :param FXEXP: GENERAL Parameter
    :return:
        EDIR: Surface soil layer direct soil evaporation
    """
    SRATIO = (SMC - SMCDRY) / (SMCMAX - SMCDRY)
    SRATIO = torch.clamp(SRATIO, min=1e-9, max=None)  # Prevents numerical instability
    FX = torch.where(torch.gt(SRATIO, 0.0), torch.clamp(SRATIO ** FXEXP, min=0.0, max=1.0), tensor(0.0))
    return FX * (1.0 - SHDFAC) * ETP1


def TRANSP(STYPE, ETP1, SMC, CMC, SHDFAC, CMCMAX, PC, CFACTR, NROOT: int,
           RTDIS) -> Tensor:
    """
    CALCULATE TRANSPIRATION FOR THE VEG CLASS.
    :param STYPE:
    :param ETP1:
    :param SMC:
    :param CMC:
    :param SHDFAC:
    :param CMCMAX:
    :param PC:
    :param CFACTR:
    :param NROOT:
    :param RTDIS:
    :return:
        ET
    """
    NSOIL = SMC.size(0)
    exponent_part = torch.clamp(CMC / CMCMAX, min=1e-9, max=None)
    ETP1A = torch.where(CMC != 0.0, SHDFAC * PC * ETP1 * (1.0 - exponent_part ** CFACTR), SHDFAC * PC * ETP1)
    ET = torch.zeros_like(SMC)
    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE[:NROOT], torch.arange(NSOIL),
                                                                                    True)
    GX = torch.clamp((SMC[:NROOT] - SMCWLT) / (SMCREF - SMCWLT), min=0.0, max=1.0)
    SGX = torch.sum(GX)
    SGX = SGX / NROOT
    RTX = RTDIS[:NROOT] + GX[:NROOT] - SGX
    GX = GX * torch.clamp(RTX, min=0.0)
    DENOM = torch.sum(GX)
    DENOM = torch.where(Tensor(DENOM <= 0.0), tensor(1), DENOM)
    ET[:NROOT] = ETP1A * GX / DENOM
    return ET



def EVAPO(SMC, CMC, ETP1, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC, CMCMAX,
          SMCDRY, CFACTR, NROOT: int, RTDIS, FXEXP):
    """
    CALCULATE SOIL MOISTURE FLUX.  THE SOIL MOISTURE CONTENT (SMC - A PER
    UNIT VOLUME MEASUREMENT) IS A DEPENDENT VARIABLE THAT IS UPDATED WITH
    PROGNOSTIC EQNS. THE CANOPY MOISTURE CONTENT (CMC) IS ALSO UPDATED.
    FROZEN GROUND VERSION:  NEW STATES ADDED: SH2O, AND FROZEN GROUND
    CORRECTION FACTOR, FRZFACT AND PARAMETER SLOPE.

    :param SMC:
    :param CMC:
    :param ETP1:
    :param DT:
    :param SH2O:
    :param SMCMAX:
    :param PC:
    :param STYPE:
    :param SHDFAC:
    :param CMCMAX:
    :param SMCDRY:
    :param CFACTR:
    :param NROOT:
    :param RTDIS:
    :param FXEXP:
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
    EDIR = torch.where(condition1 & condition2, DEVAP(ETP1, SMC[0], SHDFAC, SMCMAX, SMCDRY, FXEXP), tensor(0))

    ET = torch.where(condition1 & condition3,
                     TRANSP(STYPE, ETP1, SH2O, CMC, SHDFAC, CMCMAX, PC, CFACTR, NROOT, RTDIS),
                     torch.zeros_like(SH2O))
    # total transpiration
    ETT = torch.sum(ET)
    # Constraints on exponential operations
    exponent_part = torch.clamp(CMC / CMCMAX, min=1e-9, max=None)
    # canopy evaporation
    EC = torch.where(condition1 & condition3 & condition4, SHDFAC * (exponent_part ** CFACTR) * ETP1, tensor(0.0))
    CMC2MS = CMC / DT
    EC = torch.min(CMC2MS, EC)
    # total evapotranspiration
    ETA1 = EDIR + ETT + EC
    return ETA1, EDIR, ETT, EC, ET





def WDFCND_NY06(SMC, SMCMAX, BEXP, DKSAT, DWSAT, SICE):
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


def FAC2MIT(SMCMAX: torch.Tensor) -> torch.Tensor:
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


def SRT(STYPE, EDIR, ET, SH2O, SH2OA, PCPDRP, ZSOIL, DT, SLOPE, KDT, FRZX, SICE, CMC, CMCMAX, RHSCT):
    """
    CALCULATE THE RIGHT HAND SIDE OF THE TIME TENDENCY TERM OF THE SOIL
    WATER DIFFUSION EQUATION.  ALSO TO COMPUTE ( PREPARE ) THE MATRIX
    COEFFICIENTS FOR THE TRI-DIAGONAL MATRIX OF THE IMPLICIT TIME SCHEME.
    :param STYPE:
    :param EDIR:
    :param ET:
    :param SH2O:
    :param SH2OA:
    :param PCPDRP:
    :param ZSOIL:
    :param DT:
    :param SLOPE:
    :param KDT:
    :param FRZX:
    :param SICE:
    :return: RUNOFF1, RUNOFF2, AI, BI, CI, RHSTT
    """
    # OPT_INF = 2
    NSOIL = SH2O.size(0)
    # SICEMAX = torch.max(SICE)
    # PDDUM = PCPDRP
    # RUNOFF1 = tensor(0.0)
    RUNOFF2 = tensor(0.0)
    DENOM2 = torch.zeros_like(SH2O)
    DENOM2[0] = -ZSOIL[0]
    DENOM2[1:] = ZSOIL[:-1] - ZSOIL[1:]

    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE[:NSOIL], torch.arange(NSOIL),
                                                                                    True)
    NROOT = 8
    SMCAV = SMCMAX[0] - SMCWLT[0]
    DMAX = DENOM2[:NROOT] * SMCAV * (1 - (SH2OA[:NROOT] + SICE[:NROOT] - SMCWLT[:NROOT]) / SMCAV)
    DD = torch.sum(DMAX)
    DDT = DD * (1.0 - torch.exp(- KDT * DT / 86400.0))
    PX = torch.clamp(PCPDRP * DT, min=0.0)

    WDF2, WCND2 = WDFCND_NY06(SH2OA + SICE, SMCMAX, BEXP, DKSAT, DWSAT, SICE)

    INFMAX = torch.clamp((PX * (DDT / (PX + DDT))) / DT, min=WCND2[0], max=PX / DT)
    RUNOFF1 = torch.where(torch.gt(PCPDRP, INFMAX), PCPDRP - INFMAX, tensor(0.0))
    PDDUM = torch.where(torch.gt(PCPDRP, INFMAX), INFMAX, PCPDRP)

    #####################################################

    # Vector operations are used after optimization
    DDZ2 = torch.zeros(NSOIL - 1)
    AI = torch.zeros(NSOIL)
    CI = torch.zeros(NSOIL)
    RHSTT = torch.zeros(NSOIL)
    SLOPE_tensor = torch.ones(NSOIL - 1)
    delta_H2O = torch.zeros(NSOIL)
    SLOPE_tensor[-1] = SLOPE
    delta_H2O[:-1] = (SH2O[:-1] - SH2O[1:])
    DDZ2[0] = (0 - ZSOIL[1]) * 0.5
    DDZ2[1:] = (ZSOIL[:-2] - ZSOIL[2:]) * 0.5
    AI[1:] = -WDF2[:-1] / (DDZ2 * DENOM2[1:])
    CI[:-1] = -WDF2[:-1] / (DDZ2 * DENOM2[:-1])
    BI = -(AI + CI)
    RHSTT[0] = CI[0] * delta_H2O[0] + (WCND2[0] - PDDUM + EDIR + ET[0]) / (-DENOM2[0])
    RHSTT[1:] = -AI[1:] * delta_H2O[:-1] + CI[1:] * delta_H2O[1:] + (WCND2[1:] * SLOPE_tensor - WCND2[:-1] + ET[1:]) / (
        -DENOM2[1:])
    ######################################################
    AI = AI.detach()
    BI = BI.detach()
    CI = CI.detach()

    RHSTT = RHSTT * DT
    AI = AI * DT
    BI = 1 + BI * DT
    CI = CI * DT

    CO_Matrix = torch.diag(AI[1:], -1) + torch.diag(BI) + torch.diag(CI[:-1], 1)
    P = torch.linalg.solve(CO_Matrix, RHSTT)
    ###########################################################
    SMCMAX = SMCMAX.detach()
    PLUS = SMCMAX - (SH2O + P + SICE)
    if (PLUS < 0).any():
        DDZ = torch.zeros(NSOIL)
        DDZ[0] = -ZSOIL[0]
        DDZ[1:] = ZSOIL[:-1] - ZSOIL[1:]
        WPLUS = torch.zeros(NSOIL + 1)
        for K in range(1, NSOIL + 1):
            WPLUS[K] = torch.clamp_min_(
                ((SH2O[K - 1] + P[K - 1] + SICE[K - 1] + WPLUS[K - 1] / DDZ[K - 1]) - SMCMAX[K - 1]) * DDZ[K - 1], 0)
        SH2OOUT = torch.clamp(SH2O + P + WPLUS[:-1] / DDZ, min=tensor(0.0), max=SMCMAX - SICE)
        SMC = torch.clamp(SH2O + P + WPLUS[:-1] / DDZ + SICE, tensor(0.02), SMCMAX)
        RUNOFF3 = WPLUS[-1]
    else:
        SH2OOUT = torch.clamp(SH2O + P, min=tensor(0.0), max=SMCMAX - SICE)
        SMC = torch.clamp(SH2O + P + SICE, tensor(0.02), SMCMAX)
        RUNOFF3 = tensor(0.0)
    ##########################
    CMC = CMC + DT * RHSCT
    CMC = torch.where(CMC < 1E-20, tensor(0.0), CMC)
    CMC = torch.clamp(CMC, tensor(0.0), CMCMAX)
    SMC = SMC.detach()

    return RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMC, CMC


def REDPRM(VEGTYP: int, SOILTYP: int, SLOPETYP: int, SLDPTH, ZSOIL, SHDFAC):
    """
    READ THE PARAMETERS
    :param VEGTYP:  Vegetation type
    :param SOILTYP: First layer soil type
    :param SLOPETYP: Slope type
    :param SLDPTH:  Thickness of each layer of soil
    :param ZSOIL:  Bottom depth of each layer of soil (negative)
    :param SHDFAC: Areal fractional coverage of green vegetation ( fraction [0.0-1.0] ).
    :return:
        CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA, SHDFAC, RSMIN, RGL, HS, ZBOT,
        FRZX, PSISAT, SLOPE, SNUP, SALP, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, FXEXP,
        RTDIS, NROOT, CZIL, LAIMIN, LAIMAX, EMISSMIN, EMISSMAX, ALBEDOMIN, ALBEDOMAX, Z0MIN,
        Z0MAX, CSOIL, PTU, LVCOEF
    """
    CSOIL = Tensor(gen_parameters['CSOIL_DATA'])
    BEXP, SMCMAX, DKSAT, PSISAT = Soil_Parameter.get_by_index(SOILTYP)
    QUARTZ = Soil_Parameter.get_QTZ_by_index(SOILTYP)
    F1 = torch.log10(PSISAT) + BEXP * torch.log10(SMCMAX) + 2.0
    REFSMC1 = SMCMAX * (5.79E-9 / DKSAT) ** (1 / (2 * BEXP + 3))
    SMCREF = REFSMC1 + 1. / 3. * (SMCMAX - REFSMC1)
    WLTSMC1 = SMCMAX * (200. / PSISAT) ** (-1. / BEXP)
    SMCWLT = 0.5 * WLTSMC1
    DWSAT = BEXP * DKSAT * PSISAT / SMCMAX
    SMCDRY = SMCWLT
    ZBOT = gen_parameters['ZBOT_DATA']
    SALP = gen_parameters['SALP_DATA']
    SBETA = gen_parameters['SBETA_DATA']
    REFDK = gen_parameters['REFDK_DATA']
    FRZK = gen_parameters['FRZK_DATA']
    FXEXP = gen_parameters['FXEXP_DATA']
    REFKDT = gen_parameters['REFKDT_DATA']
    PTU = tensor(0.0)
    KDT = REFKDT * DKSAT / REFDK
    CZIL = gen_parameters['CZIL_DATA']
    SLOPE = gen_parameters['SLOPE_DATA'][int(SLOPETYP) - 1]
    LVCOEF = gen_parameters['LVCOEF_DATA']

    FRZFACT = (SMCMAX / SMCREF) * (0.412 / 0.468)
    FRZX = FRZK * FRZFACT

    TOPT = gen_parameters['TOPT_DATA']
    CMCMAX = gen_parameters['CMCMAX_DATA']
    CFACTR = gen_parameters['CFACTR_DATA']
    RSMAX = gen_parameters['RSMAX_DATA']
    NROOT = int(veg_parameter.at[VEGTYP, 'NROOT'])
    if NROOT < 1:
        NROOT = 1
    SNUP = torch.tensor(veg_parameter.at[VEGTYP, 'SNUP'])
    RSMIN = torch.tensor(veg_parameter.at[VEGTYP, 'RS'])
    RGL = torch.tensor(veg_parameter.at[VEGTYP, 'RGL'])
    HS = torch.tensor(veg_parameter.at[VEGTYP, 'HS'])
    EMISSMIN = torch.tensor(veg_parameter.at[VEGTYP, 'EMISSMIN'])
    EMISSMAX = torch.tensor(veg_parameter.at[VEGTYP, 'EMISSMAX'])
    LAIMIN = torch.tensor(veg_parameter.at[VEGTYP, 'LAIMIN'])
    LAIMAX = torch.tensor(veg_parameter.at[VEGTYP, 'LAIMAX'])
    Z0MIN = torch.tensor(veg_parameter.at[VEGTYP, 'Z0MIN'])
    Z0MAX = torch.tensor(veg_parameter.at[VEGTYP, 'Z0MAX'])
    ALBEDOMIN = torch.tensor(veg_parameter.at[VEGTYP, 'ALBEDOMIN'])
    ALBEDOMAX = torch.tensor(veg_parameter.at[VEGTYP, 'ALBEDOMAX'])

    SHDFAC = torch.where(tensor(VEGTYP == BARE), tensor(0.0), SHDFAC)
    RTDIS = - SLDPTH[:NROOT] / ZSOIL[NROOT - 1]

    return (CFACTR, CMCMAX, RSMAX, TOPT, REFKDT, KDT, SBETA, SHDFAC, RSMIN, RGL, HS, ZBOT,
            FRZX, PSISAT, SLOPE, SNUP, SALP, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ, FXEXP,
            RTDIS, NROOT, CZIL, LAIMIN, LAIMAX, EMISSMIN, EMISSMAX, ALBEDOMIN, ALBEDOMAX, Z0MIN,
            Z0MAX, CSOIL, PTU, LVCOEF)


def CSNOW(DSNOW):
    """
    CALCULATE SNOW TERMAL CONDUCTIVITY
    :param DSNOW: Snow density
    :return:
        SNOW TERMAL CONDUCTIVITY
    """
    UNIT = 0.11631
    C = 0.328 * 10 ** (2.25 * DSNOW)
    return 2.0 * UNIT * C


def SNFRAC(SNEQV, SNUP, SALP):
    """
    CALCULATE SNOW FRACTION (0 -> 1)
    :param SNEQV:
    :param SNUP:
    :param SALP:
    :return:
        SNCOVR
    """
    RSNOW = SNEQV / SNUP
    SNCOVR = torch.where(Tensor(SNEQV < SNUP),
                         1 - (torch.exp(- SALP * RSNOW) - RSNOW * torch.exp(- SALP)),
                         tensor(1.0))
    return SNCOVR


def ALCALC(ALB, SNOALB, EMBRD, SNCOVR, DT, SNOWNG, SNOTIME1, LVCOEF):
    """
    CALCULATE ALBEDO INCLUDING SNOW EFFECT (0 -> 1)
    :param ALB: SNOWFREE ALBEDO
    :param SNOALB: MAXIMUM (DEEP) SNOW ALBEDO
    :param EMBRD: BACKGROUND EMISSIVITY
    :param SNCOVR: FRACTIONAL SNOW COVER
    :param DT: TIMESTEP
    :param SNOWNG: SNOW FLAG
    :param SNOTIME1: SNOW
    :param LVCOEF:
    :return:
        ALBEDO, EMISSI, SNOTIME1
    """
    SNACCA = 0.94
    SNACCB = 0.58
    EMISSI = EMBRD + SNCOVR * (EMISSI_S - EMBRD)
    SNOALB1 = SNOALB + LVCOEF * (0.85 - SNOALB)
    SNOTIME1 = torch.where(SNOWNG, tensor(0.0), SNOTIME1 + DT)
    SNOALB2 = torch.where(SNOWNG, SNOALB1, SNOALB1 * (SNACCA ** ((SNOTIME1 / 86400.0) ** SNACCB)))
    SNOALB2 = torch.max(SNOALB2, ALB)
    ALBEDO = ALB + SNCOVR * (SNOALB2 - ALB)
    ALBEDO = torch.clamp(ALBEDO, max=SNOALB2)
    return ALBEDO, EMISSI, SNOTIME1


@torch.jit.script
def TDFCND_C05_Tensor(SMC, QZ, SMCMAX, SH2O, NSOILTYPE):
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
    GRAVELBEDROCKOTHER = torch.full_like(NSOILTYPE, 3)  # 默认值为3
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


def SNOW_NEW(TEMP, NEWSN, SNOWH, SNDENS):
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




def SNOWZ0(SNCOVR, Z0BRD, SNOWH):
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





def WDFCND(SMC, SMCMAX, BEXP, DKSAT, DWSAT, SICEMAX):
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



def TMPAVG(TUP, TM, TDN, ZSOIL):
    """
    CALCULATE SOIL LAYER AVERAGE TEMPERATURE (TAVG) IN FREEZING/THAWING
    LAYER USING UP, DOWN, AND MIDDLE LAYER TEMPERATURES (TUP, TDN, TM),
    WHERE TUP IS AT TOP BOUNDARY OF LAYER, TDN IS AT BOTTOM BOUNDARY OF LAYER.
    TM IS LAYER PROGNOSTIC STATE TEMPERATURE.
    :param TUP:
    :param TM:
    :param TDN:
    :param ZSOIL:
    :return: TSVG
    """
    T0 = 273.15

    DZ = torch.zeros_like(TUP)
    DZ[0] = -ZSOIL[0]
    DZ[1:] = ZSOIL[:-1] - ZSOIL[1:]

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



def TBND(TU, TB, ZSOIL, ZBOT):
    """
    CALCULATE TEMPERATURE ON THE BOUNDARY OF THE LAYER BY INTERPOLATION OF
    THE MIDDLE LAYER TEMPERATURES
    :param TU:
    :param TB:
    :param ZSOIL:
    :param ZBOT:
    :return: TBAND
    """
    delta = torch.zeros_like(TU)
    delta[0] = (0 - ZSOIL[0]) / (0 - ZSOIL[1])
    delta[1:-1] = (ZSOIL[0:-2] - ZSOIL[1:-1]) / (ZSOIL[0:-2] - ZSOIL[2:])
    delta[-1] = (ZSOIL[-2] - ZSOIL[-1]) / (ZSOIL[-2] - (2 * ZBOT - ZSOIL[-1]))
    # TU + TB - TU * delta
    return TU + (TB - TU) * delta



def FRH2O_tensor(TKELV, SMC, SH2O, SMCMAX, BEXP, PSIS):
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
            print("error")
            return SH2O
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


def FRH2O_NY06(TKELV, SMCMAX, BEXP, PSIS, SMC):
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



def SNOWPACK(ESD, DTSEC, SNOWH, SNDENS, TSNOW, TSOIL):
    """
    CALCULATE COMPACTION OF SNOWPACK UNDER CONDITIONS OF INCREASING SNOW
    DENSITY, AS OBTAINED FROM AN APPROXIMATE SOLUTION OF E. ANDERSON'S
    DIFFERENTIAL EQUATION (3.29), NOAA TECHNICAL REPORT NWS 19, BY VICTOR
    KOREN, 03/25/95.
    :param ESD:
    :param DTSEC:
    :param SNOWH:
    :param SNDENS:
    :param TSNOW:
    :param TSOIL:
    :return:
    """
    C1 = 0.01
    C2 = 21.0
    ESDC = ESD * 100.
    DTHR = DTSEC / 3600.
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


def SNKSRC(TAVG, SMC, SH2O, ZSOIL, SMCMAX, PSISAT, BEXP, DT, QTOT):
    """
    CALCULATE SINK/SOURCE TERM OF THE TERMAL DIFFUSION EQUATION. (SH2O) IS
    AVAILABLE LIQUED WATER.
    :param TAVG:
    :param SMC:
    :param SH2O:
    :param ZSOIL:
    :param SMCMAX:
    :param PSISAT:
    :param BEXP:
    :param DT:
    :param QTOT:
    :return:
        TSNSR, XH2O
    """
    DH2O = 1.0000E3
    HLICE = 3.3350E5
    DZ = torch.zeros(ZSOIL.size())
    DZ[0] = -ZSOIL[0]
    DZ[1:] = ZSOIL[:-1] - ZSOIL[1:]
    FREE = FRH2O_tensor(TAVG.detach(), SMC, SH2O, SMCMAX, BEXP, PSISAT)
    FREE = FREE - SH2O.detach() + SH2O
    XH2O = SH2O + QTOT.detach() * DT / (DH2O * HLICE * DZ)
    XH2O = torch.where(torch.lt(XH2O, SH2O) & torch.lt(XH2O, FREE), torch.min(FREE, SH2O), XH2O)
    XH2O = torch.where(torch.gt(XH2O, SH2O) & torch.gt(XH2O, FREE), torch.max(FREE, SH2O), XH2O)
    XH2O = torch.clamp(XH2O, tensor(0.0), SMC.detach())
    TSNSR = -DH2O * HLICE * DZ * (XH2O - SH2O).detach() / DT
    return TSNSR, XH2O



def SHFLX(STYPE, STC, SMC, DT, YY, ZZ1, ZSOIL, TBOT, ZBOT, SH2O, DF1, CSOIL):
    """
    UPDATE THE TEMPERATURE STATE OF THE SOIL COLUMN BASED ON THE THERMAL
    DIFFUSION EQUATION AND UPDATE THE FROZEN SOIL MOISTURE CONTENT BASED
    ON THE TEMPERATURE.
    :param STYPE:
    :param STC:
    :param SMC:
    :param DT:
    :param YY:
    :param ZZ1:
    :param ZSOIL:
    :param TBOT:
    :param ZBOT:
    :param SH2O:
    :param DF1:
    :param CSOIL:
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
    CSOIL_LOC = CSOIL

    # Vector operations are used after optimization
    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE, torch.arange(NSOIL), True)
    DF1N = TDFCND_C05_Tensor(SMC.detach(), QUARTZ, SMCMAX, SH2O.detach(), STYPE)
    SICE = SMC - SH2O
    DF1N[0] = DF1
    DENOM2 = torch.zeros(NSOIL)  # ΔZk
    DDZ2 = torch.zeros(NSOIL - 1)  # ΔZk_tilde
    AI = torch.zeros(NSOIL)  # subdiagonal elements
    CI = torch.zeros(NSOIL)  # superdiagonal elements
    BI = torch.zeros(NSOIL)  # diagonal elements
    RHSTS = torch.zeros(NSOIL)  # Right hand
    delta_STC = torch.zeros(NSOIL)
    delta_STC[:-1] = (STC[:-1] - STC[1:])
    delta_STC[-1] = STC[-1] - TBOT
    # Calculation Graph of Separating Temperature and Soil Water
    HCPCT = SH2O.detach() * CH2O + (1.0 - SMCMAX) * CSOIL_LOC + (SMCMAX - SMC.detach()) * CAIR + SICE.detach() * CICE
    # HCPCT = SH2O * CH2O + (1.0 - SMCMAX) * CSOIL_LOC + (SMCMAX - SMC) * CAIR + SICE * CICE
    DENOM2[0] = -ZSOIL[0]
    DENOM2[1:] = ZSOIL[:-1] - ZSOIL[1:]
    DDZ2[0] = (0 - ZSOIL[1]) * 0.5
    DDZ2[1:] = (ZSOIL[:-2] - ZSOIL[2:]) * 0.5
    AI[1:] = -DF1N[:-1] / (DENOM2[1:] * DDZ2 * HCPCT[1:])
    CI[:-1] = -DF1N[:-1] / (DENOM2[:-1] * DDZ2 * HCPCT[:-1])
    BI[0] = -CI[0] + DF1N[0] / (0.5 * ZSOIL[0] * ZSOIL[0] * HCPCT[0] * ZZ1)
    BI[1:] = -(AI[1:] + CI[1:])
    SSOIL = DF1N[0] * (STC[0] - YY) / (0.5 * ZSOIL[0] * ZZ1)
    RHSTS[0] = (SSOIL - delta_STC[0] * DF1N[0] / DDZ2[0]) / (DENOM2[0] * HCPCT[0])
    RHSTS[1:-1] = (-AI[1:-1] * delta_STC[:-2] + CI[1:-1] * delta_STC[1:-1])
    RHSTS[-1] = -AI[-1] * delta_STC[-2] - (DF1N[-1] * delta_STC[-1]) / (
            DENOM2[-1] * HCPCT[-1] * (.5 * (ZSOIL[-2] + ZSOIL[-1]) - ZBOT))
    QTOT = RHSTS * DENOM2 * HCPCT

    TB = torch.zeros(NSOIL)
    TB[:-1] = STC[1:]
    TB[-1] = TBOT
    TDN = TBND(STC, TB, ZSOIL, ZBOT)
    TUP = torch.zeros(NSOIL)
    TUP[0] = (YY + (ZZ1 - 1) * STC[0]) / ZZ1  # TSURF
    TUP[1:] = TDN[:-1]
    TAVG = TMPAVG(TUP, STC, TDN, ZSOIL)  # Average Temperature
    TSNSR, SH2O_New = SNKSRC(TAVG, SMC, SH2O, ZSOIL, SMCMAX, PSISAT, BEXP, DT, QTOT)  # phase change
    condition = torch.gt(SICE, 0) | torch.lt(STC, T0) | torch.lt(TUP, T0) | torch.lt(TDN,
                                                                                     T0)  # Phase change and calculate supercooled liquid water
    TSNSR = torch.where(condition, TSNSR, 0)  # The diffusion equation right-hand term source and sink
    SH2O_New = torch.where(condition, SH2O_New, SH2O)
    RHSTS = RHSTS + TSNSR / (DENOM2 * HCPCT)

    RHSTS = RHSTS * DT
    AI = AI * DT
    BI = 1 + BI * DT
    CI = CI * DT

    CO_Matrix = torch.diag(AI[1:], -1) + torch.diag(BI) + torch.diag(CI[:-1], 1)
    P = torch.linalg.solve(CO_Matrix, RHSTS)
    STC = STC + P

    T1 = (YY + (ZZ1 - 1.0) * STC[0]) / ZZ1
    SSOIL = DF1 * (STC[0] - T1) / (0.5 * ZSOIL[0])
    return STC, T1, SSOIL, SH2O_New



def SMFLX(STYPE, SMC, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE, KDT, FRZFACT,
          SHDFAC, CMCMAX, EDIR, EC, ET):
    """
    CALCULATE SOIL MOISTURE FLUX.  THE SOIL MOISTURE CONTENT (SMC - A PER
    UNIT VOLUME MEASUREMENT) IS A DEPENDENT VARIABLE THAT IS UPDATED WITH
    PROGNOSTIC EQNS. THE CANOPY MOISTURE CONTENT (CMC) IS ALSO UPDATED.
    FROZEN GROUND VERSION:  NEW STATES ADDED: SH2O, AND FROZEN GROUND
    CORRECTION FACTOR, FRZFACT AND PARAMETER SLOPE.
    :param STYPE:
    :param SMC:
    :param CMC:
    :param DT:
    :param PRCP1:
    :param ZSOIL:
    :param SH2O:
    :param SLOPE:
    :param KDT:
    :param FRZFACT:
    :param SHDFAC:
    :param CMCMAX:
    :param EDIR:
    :param EC:
    :param ET:
    :return:
        SH2OOUT, SMCOUT, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP
    """
    NSOIL = SMC.size(0)
    RHSCT = SHDFAC * PRCP1 - EC
    # DRIP = tensor(0.0)
    # FAC2 = tensor(0.0)
    TRHSCT = DT * RHSCT
    EXCESS = CMC + TRHSCT
    DRIP = torch.where(torch.gt(EXCESS, CMCMAX), EXCESS - CMCMAX, tensor(0.0))

    PCPDRP = (1. - SHDFAC) * PRCP1 + DRIP / DT
    SICE = (SMC - SH2O)
    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE, torch.arange(NSOIL),
                                                                                    required_grad=False)
    FAC2 = torch.max(SH2O.detach() / SMCMAX)
    SMCMAX = SMCMAX[0]
    FLIMIT = FAC2MIT(SMCMAX)
    """Kalnay E, Kanamitsu M. Time Schemes for Strongly Nonlinear Damping Equations[J]. Monthly Weather Review, 1988, 116(10): 1945-1958."""
    if (PCPDRP * DT) > (0.0001 * 1000.0 * (- ZSOIL[0]) * SMCMAX) or FAC2 > FLIMIT:
        RUNOFF1, RUNOFF2, RUNOFF3, SH2OFG, SMCFG, DUMMY = SRT(STYPE, EDIR, ET, SH2O, SH2O, PCPDRP, ZSOIL, DT, SLOPE,
                                                              KDT,
                                                              FRZFACT, SICE, CMC, CMCMAX, RHSCT)
        SH2OA = 0.5 * (SH2O.detach() + SH2OFG)
        RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMCOUT, CMC = SRT(STYPE, EDIR, ET, SH2O, SH2OA, PCPDRP, ZSOIL, DT, SLOPE,
                                                              KDT,
                                                              FRZFACT, SICE, CMC, CMCMAX, RHSCT)
    else:
        RUNOFF1, RUNOFF2, RUNOFF3, SH2OOUT, SMCOUT, CMC = SRT(STYPE, EDIR, ET, SH2O, SH2O, PCPDRP, ZSOIL, DT, SLOPE,
                                                              KDT,
                                                              FRZFACT, SICE, CMC, CMCMAX, RHSCT)
    return SH2OOUT, SMCOUT, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP





def PENMAN(SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT, SNOWNG, FRZGRA, DQSDT2,
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
    # if FRZGRA:
    #     FLX2 = -LSUBF * PRCP
    #     FNET = FNET - FLX2

    RAD = FNET / RCH + TH2 - SFCTMP
    A = ELCP1 * (Q2SAT - Q2)
    EPSCA = (A * RR + RAD * DELTA) / (DELTA + RR)
    ETP = EPSCA * RCH / LVS
    return EPSCA, ETP, RCH, RR, FLX2, T24



def CANRES(SOLAR, CH, SFCTMP, Q2, SFCPRS, SMC, ZSOIL, STYPE, RSMIN, Q2SAT, DQSDT2,
           TOPT, RSMAX, RGL, HS, XLAI, EMISSI, NROOT: int):
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
    :param ZSOIL: SOIL DEPTH (NEGATIVE SIGN, AS IT IS BELOW GROUND)
    :param STYPE: SOIL TYPE
    :param RSMIN:  VEGETATION PARAMETERS
    :param Q2SAT: SATURATION AIR HUMIDITY AT 1ST LEVEL ABOVE GROUND
    :param DQSDT2: SLOPE OF SATURATION HUMIDITY FUNCTION WRT TEMP
    :param TOPT:  VEGETATION PARAMETERS
    :param RSMAX: VEGETATION PARAMETERS
    :param RGL: VEGETATION PARAMETERS
    :param HS: VEGETATION PARAMETERS
    :param XLAI: VEGETATION PARAMETERS
    :param EMISSI:  Emissivity ( fraction )
    :return:
        PC, RC, RCS, RCT, RCQ, RCSOIL
    """
    CP = 1004.5
    RD = 287.04
    SIGMA = 5.67E-8
    SLV = 2.501000E6

    FF = 0.55 * 2.0 * SOLAR / (RGL * XLAI)
    RCS = (FF + RSMIN / RSMAX) / (1.0 + FF)
    RCS = torch.clamp_min(RCS, 0.0001)

    RCT = 1.0 - 0.0016 * ((TOPT - SFCTMP) ** 2.0)
    RCT = torch.clamp_min(RCT, 0.0001)

    RCQ = 1.0 / (1.0 + HS * (Q2SAT - Q2))
    RCQ = torch.clamp_min(RCQ, 0.01)

    PSISAT, BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, SMCREF, SMCDRY, F1, QUARTZ = REDSTP(STYPE[:NROOT], torch.arange(NROOT),
                                                                                    True)
    GX = torch.clamp((SMC[:NROOT] - SMCWLT) / (SMCREF - SMCWLT), min=0.0, max=1.0)
    delta = torch.zeros_like(ZSOIL)
    delta[0] = ZSOIL[0]
    delta[1:] = ZSOIL[1:] - ZSOIL[:-1]
    PART = delta[:NROOT] * GX / ZSOIL[NROOT - 1]

    RCSOIL = PART.sum()

    RCSOIL = torch.clamp_min(RCSOIL, 0.0001)

    RC = RSMIN / (XLAI * RCS * RCT * RCQ * RCSOIL)

    RR = (4. * EMISSI * SIGMA * RD / CP) * (SFCTMP ** 4.) / (SFCPRS * CH) + 1.0
    DELTA = (SLV / CP) * DQSDT2

    PC = (RR + DELTA) / (RR * (1. + RC * CH) + DELTA)
    return PC, RC, RCS, RCT, RCQ, RCSOIL



def NOPAC(STYPE, ETP, PRCP, SMC, SMCMAX, SMCDRY, CMC, CMCMAX, DT, SHDFAC, SBETA,
          SFCTMP, T24, TH2, FDOWN, EMISSI, STC, EPSCA, PC, RCH, RR, CFACTR, SH2O, SLOPE, KDT, FRZFACT, ZSOIL, TBOT,
          ZBOT, NROOT: int, RTDIS, QUARTZ, FXEXP, CSOIL, lstm_input):
    """
    CALCULATE SOIL MOISTURE AND HEAT FLUX VALUES AND UPDATE SOIL MOISTURE
    CONTENT AND SOIL HEAT CONTENT VALUES FOR THE CASE WHEN NO SNOW PACK IS
    PRESENT.
    :param STYPE:
    :param ETP:
    :param PRCP:
    :param SMC:
    :param SMCMAX:
    :param SMCDRY:
    :param CMC:
    :param CMCMAX:
    :param DT:
    :param SHDFAC:
    :param SBETA:
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
    :param CFACTR:
    :param SH2O:
    :param SLOPE:
    :param KDT:
    :param FRZFACT:
    :param ZSOIL:
    :param TBOT:
    :param ZBOT:
    :param NROOT:
    :param RTDIS:
    :param QUARTZ:
    :param FXEXP:
    :param CSOIL:
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
    ETA, EDIR, ETT, EC, ET = EVAPO(SMC, CMC, ETP1, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC, CMCMAX,
                                   SMCDRY, CFACTR, NROOT, RTDIS, FXEXP)
    DEW = torch.where(torch.gt(ETP, 0), tensor(0.0), -ETP1)
    PRCP1 = torch.where(torch.gt(ETP, 0), PRCP * 0.001, PRCP1 + DEW)
    SH2O, SMC, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP = SMFLX(STYPE, SMC, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE,
                                                            KDT, FRZFACT, SHDFAC, CMCMAX, EDIR, EC, ET)
    ETA = torch.where(torch.gt(ETP, 0), ETA * 1000, ETP)  # total evapotranspiration KG M-2 S-1
    BETA = torch.where(torch.gt(ETP, 0), ETA.detach() / ETP, torch.where(torch.lt(ETP, 0.0), 1, 0))
    EC = EC * 1000
    ET = ET * 1000
    EDIR = EDIR * 1000
    DF1 = TDFCND_C05_Tensor(SMC[0].detach(), QUARTZ, SMCMAX, SH2O[0].detach(), STYPE[0])
    DF1 = DF1 * torch.exp(SBETA * SHDFAC)
    YYNUM = FDOWN - EMISSI * SIGMA * T24
    YY = SFCTMP + (YYNUM / RCH + TH2 - SFCTMP - BETA * EPSCA) / RR
    ZZ1 = DF1 / (-0.5 * ZSOIL[0] * RCH * RR) + 1.0
    STC, T1, SSOIL, SH2O = SHFLX(STYPE, STC, SMC, DT, YY, ZZ1, ZSOIL, TBOT, ZBOT, SH2O, DF1, CSOIL)
    FLX1 = CPH2O * PRCP * (T1 - SFCTMP)
    FLX3 = tensor(0.0)
    return STC, SMC, SH2O, DEW, DRIP, EC, EDIR, ETA, ET, ETT, RUNOFF1, RUNOFF2, RUNOFF3, SSOIL, CMC, BETA, T1, FLX1, FLX3

def SNOPAC(STYPE, ETP, PRCP, PRCPF, SNOWNG, SMC, SMCMAX, SMCDRY, CMC, CMCMAX, DT, DF1,
           T1, SFCTMP, T24, TH2, FDOWN, STC, PC, RCH, RR, CFACTR, SNCOVR,
           ESD: Tensor, SNDENS, SNOWH, SH2O, SLOPE, KDT, FRZFACT, ZSOIL, TBOT, ZBOT, SHDFAC, NROOT: int, RTDIS,
           FXEXP,
           CSOIL, FLX2, EMISSI, RIBB, lstm_input):
    """
    CALCULATE SOIL MOISTURE AND HEAT FLUX VALUES & UPDATE SOIL MOISTURE
    CONTENT AND SOIL HEAT CONTENT VALUES FOR THE CASE WHEN A SNOW PACK IS
    PRESENT.
    :param STYPE:
    :param ETP:
    :param PRCP:
    :param PRCPF:
    :param SNOWNG:
    :param SMC:
    :param SMCMAX:
    :param SMCDRY:
    :param CMC:
    :param CMCMAX:
    :param DT:
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
    :param CFACTR:
    :param SNCOVR:
    :param ESD:
    :param SNDENS:
    :param SNOWH:
    :param SH2O:
    :param SLOPE:
    :param KDT:
    :param FRZFACT:
    :param ZSOIL:
    :param TBOT:
    :param ZBOT:
    :param SHDFAC:
    :param NROOT:
    :param RTDIS:
    :param FXEXP:
    :param CSOIL:
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
        ESNOW2 = ETP1 * DT
        ETANRG = ETP * ((1. - SNCOVR) * LSUBC + SNCOVR * LSUBS)
    else:
        ETP1 = ETP * 0.001
        if SNCOVR < 1.0:
            ETNS1, EDIR, ETT, EC1, ET1 = EVAPO(SMC, CMC, ETP1, DT, SH2O, SMCMAX, PC, STYPE, SHDFAC,
                                               CMCMAX,
                                               SMCDRY, CFACTR, NROOT, RTDIS, FXEXP)
            EDIR1 = EDIR * (1. - SNCOVR)
            EDIR = EDIR1 * 1000
            ET1 = ET1 * (1. - SNCOVR)
            EC1 = EC1 * (1. - SNCOVR)
            EC = EC1 * 1000
            ET = ET1 * 1000
            ETT = ETT * (1. - SNCOVR) * 1000
            ETNS = ETNS1 * (1. - SNCOVR) * 1000

        ESNOW2 = ETP * SNCOVR * 0.001 * DT
        ETANRG = ESNOW * LSUBS + ETNS * LSUBC

    FLX1 = torch.where(SNOWNG, CPICE * PRCP * (T1 - SFCTMP),
                       torch.where(torch.gt(PRCP, 0.0), CPH2O * PRCP * (T1 - SFCTMP), 0))

    DSOIL = -(0.5 * ZSOIL[0])
    DTOT = SNOWH + DSOIL
    DENOM = 1.0 + DF1 / (DTOT * RR * RCH)
    T12A = ((FDOWN - FLX1 - FLX2 - EMISSI * SIGMA * T24) / RCH + TH2 - SFCTMP - ETANRG / RCH) / RR
    T12B = DF1 * STC[0] / (DTOT * RR * RCH)
    T12 = (SFCTMP + T12A + T12B) / DENOM
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
            SNOMLT = EX * DT
            if ESD - SNOMLT >= ESDMIN:
                ESD = ESD - SNOMLT
            else:
                EX = ESD / DT
                FLX3 = EX * 1000.0 * LSUBF
                SNOMLT = ESD
                ESD = tensor(0.0)
        PRCP1 = PRCP1 + EX
    SH2O, SMC, RUNOFF1, RUNOFF2, RUNOFF3, CMC, DRIP = SMFLX(STYPE, SMC, CMC, DT, PRCP1, ZSOIL, SH2O, SLOPE,
                                                            KDT, FRZFACT, SHDFAC, CMCMAX, EDIR1, EC1, ET1)
    ZZ1 = tensor(1.0)
    YY = STC[0] - 0.5 * SSOIL * ZSOIL[0] * ZZ1 / DF1
    # T11 = T1
    STC, T1, SSOIL, SH2O = SHFLX(STYPE, STC, SMC, DT, YY, ZZ1, ZSOIL, TBOT, ZBOT, SH2O, DF1, CSOIL)
    if ESD > 0:
        SNOWH, SNDENS = SNOWPACK(ESD, DT, SNOWH, SNDENS, T1, YY)
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

