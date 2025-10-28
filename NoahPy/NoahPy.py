'''
@Project ：NoahPy
@File    ：NoahPy.py
@Author  ：tianwb
@Date    ：2024/5/14 18:08

@Description：
This module provides a PyTorch-based differentiable implementation of the Noah LSM (Land Surface Model),
enabling gradient-based optimization and backpropagation support. It is designed for integration with 
physics-informed machine learning frameworks and research on land surface processes.
This version is based on Noah 3.4.1, and some of the physical processes have been improved,
see Wu X, Nan Z, Zhao S, et al. Spatial modeling of permafrost distribution and properties on
the Qinghai‐Tibet Plateau[J]. Permafrost and Periglacial Processes, 2018, 29(2): 86-99. doi: 10.1002/ppp.1971.
Chen H, Nan Z, Zhao L, et al. Noah Modelling of the Permafrost Distribution and Characteristics in
the West Kunlun Area, Qinghai‐Tibet Plateau, China[J]. Permafrost and Periglacial Processes, 2015, 26(2): 160-174. doi: 10.1002/ppp.1841.


@License：
Copyright (c) 2025 Tian Wenbiao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions 
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO 
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import cProfile
import io
import os
import pickle
import pstats
import subprocess
from datetime import datetime

from matplotlib import pyplot as plt


from Module_sf_noahlsm import *
from Module_sfcdif_wrf import *


def open_forcing_file(forcing_file_path):
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
        "Soil_layer_thickness": [0.045, 0.046, 0.075, 0.123, 0.204, 0.336, 0.554, 0.913, 0.904, 1, 1, 1, 1, 1, 1, 2, 2,
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
    infotext = None
    NSOIL = len(parameters['Soil_htype'])
    startdate = parameters['startdate']
    enddate = parameters['enddate']
    loop_for_a_while = parameters['loop_for_a_while']
    latitude = parameters['Latitude']
    longitude = parameters['Longitude']
    forcing_timestep = parameters['Forcing_Timestep']
    noahlsm_timestep = parameters['Noahlsm_Timestep']
    ice = parameters['Sea_ice_point']
    T1 = tensor(parameters['Skin_Temperature'])
    # STC = [tensor(item) for item in parameters['Soil_Temperature']]
    # SMC = [tensor(item) for item in parameters['Soil_Moisture']]
    # SH2O = [tensor(item) for item in parameters['Soil_Liquid']]
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
    SOILTYP = parameters['Soil_type_index']
    SLOPETYP = parameters['Slope_type_index']
    SNOALB = tensor(parameters['Max_snow_albedo'])
    ZLVL = tensor(parameters['Air_temperature_level'])
    ZLVL_WIND = tensor(parameters['Wind_level'])
    albedo_monthly = parameters['Albedo_monthly']
    shdfac_monthly = parameters['Shdfac_monthly']
    z0brd_monthly = parameters['Z0brd_monthly']
    lai_monthly = parameters['lai_monthly']
    use_urban_module = False
    ISURBAN = False
    SHDMIN = tensor(parameters['Green_Vegetation_Min'])
    SHDMAX = tensor(parameters['Green_Vegetation_Max'])
    USEMONALB = tensor(parameters['Usemonalb'])
    RDLAI2D = tensor(parameters['Rdlai2d'])
    LLANDUSE = parameters['Landuse_dataset']
    IZ0TLND = tensor(parameters['iz0tlnd'])
    sfcdif_option = parameters['sfcdif_option']
    forcing_columns_name = ['Year', 'Month', 'Day', 'Hour', 'minutes', 'windspeed', 'winddir', 'temperature',
                            'humidity', 'pressure', 'shortwave', 'longwave', 'precipitation', 'LAI', 'NDVI']
    x_target_columns = ['windspeed', 'winddir', 'temperature', 'humidity', 'pressure', 'shortwave', 'longwave',
                        'precipitation', 'LAI', 'NDVI']
    forcing_data = pd.read_csv(forcing_file_path, sep=r'\s+', names=forcing_columns_name, header=None,
                               skiprows=45)
    forcing_data['Date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day', 'Hour']])
    forcing_data.set_index('Date', inplace=True)
    forcing_data = forcing_data[x_target_columns]
    condition = forcing_data.index >= startdate
    forcing_data = forcing_data[condition]
    Date = forcing_data.index
    forcing_data = torch.tensor(forcing_data.to_numpy(), dtype=torch.float32)

    return (Date, forcing_data, output_dir, forcing_filename, infotext, NSOIL, startdate, enddate, loop_for_a_while, latitude,
            longitude,
            forcing_timestep, noahlsm_timestep, ice, T1, STC, SMC, SH2O, STYPE,
            SLDPTH, CMC, SNOWH, SNEQV, TBOT, VEGTYP, SOILTYP, SLOPETYP, SNOALB, ZLVL, ZLVL_WIND,
            albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly, use_urban_module, ISURBAN,
            SHDMIN, SHDMAX, USEMONALB, RDLAI2D, LLANDUSE, IZ0TLND, sfcdif_option)


def month_d(a12, nowdate):
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
        return a12[nowm - 1]
    elif nowd < 15:
        prevm = nowm - 1 if nowm > 1 else 12
        postm = nowm
        factor = (ndays[prevm - 1] - 15 + nowd) / ndays[prevm - 1]
    elif nowd > 15:
        prevm = nowm
        postm = nowm + 1 if nowm < 12 else 1
        factor = (nowd - 15) / ndays[prevm - 1]

    return tensor(a12[prevm - 1] * (1.0 - factor) + a12[postm - 1] * factor)


def soil_veg_gen_parm():
    """
    load lookup table parameters
    :return:
        gen_parameters, veg_parameter, soil_parameter
    """

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
        'SLOPE_DATA': [0.1, 0.6, 1.0, 0.35, 0.55, 0.8, 0.63, 0.0, 0.0]
    }
    # soil_parameter = soil_parameter.iloc[:, 1:]

    # soil_parameter = soil_parameter.loc[Soil_htype]
    return gen_parameters


def read_forcing_text(forcing, k_time):
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


def CALTMP(T1, SFCTMP, SFCPRS, ZLVL, Q2):
    TH2 = SFCTMP + (0.0098 * ZLVL)
    T1V = T1 * (1.0 + 0.61 * Q2)
    TH2V = TH2 * (1.0 + 0.61 * Q2)
    T2V = SFCTMP * (1.0 + 0.61 * Q2)
    RHO = SFCPRS / (RD * T2V)
    return TH2, T1V, TH2V, T2V, RHO




def CALHUM(SFCTMP, SFCPRS):
    A2 = 17.67
    A3 = 273.15
    A4 = 29.65
    ELWV = 2.501e6
    A23M4 = A2 * (A3 - A4)
    E0 = 611.0
    RV = 461.0
    EPSILON = 0.622
    ES = E0 * torch.exp(ELWV / RV * (1. / A3 - 1. / SFCTMP))
    Q2SAT = EPSILON * ES / (SFCPRS - (1 - EPSILON) * ES)
    DQSDT2 = Q2SAT * A23M4 / (SFCTMP - A4) ** 2
    return Q2SAT, DQSDT2





def noah_main(file_name, trained_parameter=None, lstm_model=None, output_flag=False):
    torch.set_default_dtype(torch.float32)
    import Module_sf_noahlsm
    Module_sf_noahlsm.grad_soil_parameter = trained_parameter
    Module_sf_noahlsm.lstm_model = lstm_model
    # torch.autograd.set_detect_anomaly(True)
    # torch.set_default_device('cuda')
    badval = -1.E36
    IILOC = 1  # I-index of the point being processed.
    JJLOC = 1  # J-index of the point being processed.
    PC = badval  # Plant coefficient, where PC * ETP = ETA ( Fraction [0.0-1.0] )
    RCS = badval  # Incoming solar RC factor ( dimensionless )
    RCT = badval  # Air temperature RC factor ( dimensionless )
    RCQ = badval  # Atmospheric water vapor deficit RC factor ( dimensionless )
    RCSOIL = badval  # Soil moisture RC factor ( dimensionless )
    Q1 = tensor(badval)  # Effective mixing ratio at the surface ( kg kg{-1} )
    SNOTIME1 = 0.0  # Age of the snow on the ground.

    """read forcing file"""
    (Date, forcing_data, output_dir, forcing_filename, infotext, NSOIL, startdate, enddate, loop_for_a_while,
     latitude, longitude, forcing_timestep, noahlsm_timestep, ice, T1, STC, SMC, SH2O, STYPE,
     SLDPTH, CMC, SNOWH, SNEQV, TBOT, VEGTYP, SOILTYP, SLOPETYP, SNOALB, ZLVL, ZLVL_WIND,
     albedo_monthly, shdfac_monthly, z0brd_monthly, lai_monthly, use_urban_module, ISURBAN,
     SHDMIN, SHDMAX, USEMONALB, RDLAI2D, LLANDUSE, IZ0TLND, sfcdif_option) = open_forcing_file(file_name)

    Module_sf_noahlsm.NSOIL = NSOIL
    ZSOIL = torch.zeros_like(SH2O)
    NSOIL = SH2O.size(0)
    ZSOIL[0] = -SLDPTH[0]
    for i in range(1, NSOIL):
        ZSOIL[i] = -SLDPTH[i] + ZSOIL[i - 1]

    DT = torch.tensor(noahlsm_timestep, dtype=torch.int32)
    EMISSI = 0.96
    ALBEDO = month_d(albedo_monthly, startdate)
    Z0 = month_d(z0brd_monthly, startdate)
    if sfcdif_option == 1:
        Z0BRD = Z0
    else:
        Z0BRD = torch.tensor(badval)
    CZIL = gen_parameters['CZIL_DATA']

    if sfcdif_option == 1:
        MYJSFCINIT()

    CH = 1.E-4
    CM = 1.E-4
    nowdate = startdate
    timestep = pd.Timedelta(seconds=noahlsm_timestep)
    k_time = 0

    # CMC, T1, STC, SMC, SH2O,
    # SNOWH, SNEQV,SNOTIME1,
    # PC, RCS, RCT, RCQ, RCSOIL,
    # Q1, Z0, Z0BRD, EMISSI
    out_SMC = []
    out_STC = []
    out_SH2O = []
    # out_CMC = []
    # out_T1 = []
    # out_SNOWH = []
    # out_SNEQV = []
    # out_PC = []
    # out_ETP = []
    # out_SSOIL = []
    # print("start simulate")
    while nowdate <= enddate:
        with torch.no_grad():
            SFCSPD, SFCTMP, Q2, SFCPRS, SOLDN, LONGWAVE, PRCP, SFCU, SFCV, lstm_input = read_forcing_text(forcing_data,
                                                                                                          k_time)
            FFROZP = torch.where(torch.gt(PRCP, 0) & torch.lt(SFCTMP, 273.15), torch.tensor(1), torch.tensor(0))

            TH2, T1V, TH2V, T2V, RHO = CALTMP(T1, SFCTMP, SFCPRS, ZLVL, Q2)
            Q2SAT, DQSDT2 = CALHUM(SFCTMP, SFCPRS)  # Returns Q2SAT, DQSDT2 PENMAN needed

            ALB = torch.where(USEMONALB, month_d(albedo_monthly, nowdate), torch.tensor(badval))

            XLAI = torch.where(RDLAI2D, month_d(lai_monthly, nowdate), torch.tensor(badval))
            SHDFAC = month_d(shdfac_monthly, nowdate)
            Q1 = torch.where(Q1 == badval, Q2, Q1)
            if sfcdif_option == 1:
                RIBB, CM, CH = SFCDIF_MYJ(ZLVL, ZLVL_WIND, Z0, Z0BRD, SFCPRS, T1, SFCTMP, Q1, Q2, SFCSPD, CZIL, CM,
                                          CH, IZ0TLND)
            else:
                RIBB, CH = SFCDIF_MYJ_Y08(Z0, ZLVL_WIND, ZLVL, SFCSPD, T1, SFCTMP, Q2, SFCPRS)

            SOLNET = SOLDN * (1.0 - ALBEDO)
            LWDN = LONGWAVE * EMISSI

        (CMC, T1, STC, SMC, SH2O, SNOWH, SNEQV, SNOTIME1, ALBEDO,
         PC, Q1, Z0, Z0BRD, EMISSI) = SFLX(FFROZP, DT, SLDPTH, ZSOIL, LWDN, SOLDN, SOLNET, SFCPRS, PRCP, SFCTMP, Q2, TH2,
                                              Q2SAT, DQSDT2, VEGTYP, SLOPETYP, SHDFAC, SHDMIN, SHDMAX,
                                              ALB, SNOALB, TBOT, CMC, T1, STC, SMC, SH2O, STYPE, SNOWH, SNEQV, CH,
                                              PC, XLAI, RDLAI2D, USEMONALB, SNOTIME1, RIBB, nowdate,
                                              k_time, lstm_input)

        # del (ETA, SHEAT, EC, EDIR, ET, ETT, ESNOW, DRIP, DEW,
        #      BETA, ETP, SSOIL, FLX1, FLX2, FLX3, SNOMLT, SNCOVR, RUNOFF1, RUNOFF2, RUNOFF3, RC, SOILW,
        #      SOILM, RES, QFX, Fin, FUP)

        """output"""
        out_STC.append(STC - 273.15)
        out_SH2O.append(SH2O)
        out_SMC.append(SMC)

        # CMC, T1, STC, SMC, SH2O,
        # SNOWH, SNEQV,SNOTIME1,
        # PC, RCS, RCT, RCQ, RCSOIL,
        # Q1, Z0, Z0BRD, EMISSI

        nowdate = nowdate + timestep
        #
        k_time += 1
        if k_time % 30 == 0:
            SH2O = SH2O.detach()
            STC = STC.detach()
        if k_time % 365 == 0:
            print(nowdate)

    """output"""
    condition = (Date >= startdate) & (Date <= enddate)
    if output_flag:
        SH2O_columns = [f'SH2O({i + 1})' for i in range(NSOIL)]
        STC_columns = [f'STC({i + 1})' for i in range(NSOIL)]
        SMC_columns = [f'SMC({i + 1})' for i in range(NSOIL)]
        out_columns = (STC_columns +
                       SH2O_columns +
                       SMC_columns
                       # + ["CMC", "PC", "ETP", "SNEQV", "SNOWH", "T1", "SSOIL"]
                       )
        out = torch.cat([torch.stack(out_STC),
                         torch.stack(out_SH2O),
                         torch.stack(out_SMC),
                         # torch.stack(out_CMC).reshape(-1, 1),
                         # torch.stack(out_PC).reshape(-1, 1),
                         # torch.stack(out_ETP).reshape(-1, 1),
                         # torch.stack(out_SNEQV).reshape(-1, 1),
                         # torch.stack(out_SNOWH).reshape(-1, 1),
                         # torch.stack(out_T1).reshape(-1, 1),
                         # torch.stack(out_SSOIL).reshape(-1, 1),
                         ], dim=1)
        pd.DataFrame(out.detach().numpy(), columns=out_columns,
                     index=Date[condition]).to_csv(os.path.join(output_dir, "NoahPy_output.txt"), index=True)
    return Date[condition], torch.stack(out_STC), torch.stack(out_SH2O)

