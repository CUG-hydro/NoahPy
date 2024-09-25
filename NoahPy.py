'''
@Project ：NoahPy
@File    ：NoahPy.py
@Author  ：tianwb
@Date    ：2024/5/14 18:08

This version is based on Noah 3.4.1, and some of the physical processes have been improved,
see Wu X, Nan Z, Zhao S, et al. Spatial modeling of permafrost distribution and properties on
the Qinghai‐Tibet Plateau[J]. Permafrost and Periglacial Processes, 2018, 29(2): 86-99. doi: 10.1002/ppp.1971.
Chen H, Nan Z, Zhao L, et al. Noah Modelling of the Permafrost Distribution and Characteristics in
the West Kunlun Area, Qinghai‐Tibet Plateau, China[J]. Permafrost and Periglacial Processes, 2015, 26(2): 160-174. doi: 10.1002/ppp.1841.

'''
import cProfile
import io
import os
import pickle
import pstats
import subprocess
from datetime import datetime

from matplotlib import pyplot as plt

import utils
from Module_sf_noahlsm import *
from Module_sfcdif_wrf import *
from plot_simulate import plot_timeseries
from utils import calculate_nse, NSELoss


class LSTM(nn.Module):
    def __init__(self, num_inputs, hidden_size, output_size, dropout=0.2, bias=False):
        super(LSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(num_inputs, hidden_size)
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
        intput_variable = ['windspeed', 'temperature', 'humidity', 'pressure', 'shortwave', 'longwave', 'precipitation',
                           'NDVI', 'LAI', 'ETP']
        output_variable = ['EDIR', 'EC', 'ETT']
        plant_variable = {
            'LAI': {'mean': 0, 'scale': 1},
            'NDVI': {'mean': 0, 'scale': 0.1}
        }
        with open("StandardScaler/forcing_scaler.pkl", 'rb') as f:
            norm_pickle = pickle.load(f)
            means = norm_pickle.mean_
            scales = norm_pickle.scale_
            columns = norm_pickle.feature_names_in_
            forcing_stats = {name: {'mean': mean, 'scale': scale} for name, mean, scale in zip(columns, means, scales)}
        with open("StandardScaler/simulate_data.pkl", 'rb') as f:
            denorm_pickle = pickle.load(f)
            means = denorm_pickle.mean_
            scales = denorm_pickle.scale_
            columns = denorm_pickle.feature_names_in_
            simulate_stats = {name: {'mean': mean, 'scale': scale} for name, mean, scale in zip(columns, means, scales)}
        states = pd.DataFrame({**forcing_stats, **simulate_stats, **plant_variable}).T
        self.input_mean = torch.tensor(states.loc[intput_variable, 'mean'].values, dtype=torch.float32)
        self.intput_std = torch.tensor(states.loc[intput_variable, 'scale'].values, dtype=torch.float32)
        self.output_mean = torch.tensor(states.loc[output_variable, 'mean'].values, dtype=torch.float32)
        self.output_std = torch.tensor(states.loc[output_variable, 'scale'].values, dtype=torch.float32)
        self.hx = None
        self.cx = None

    def forward(self, x, hc=None):
        x_normalized = (x - self.input_mean) / self.intput_std
        if hc is None:
            hx, cx = self.lstm_cell(x_normalized)
        else:
            hx, cx = self.lstm_cell(x_normalized, hc)
        output = self.reg(hx)
        output = output * self.output_std + self.output_mean
        return output, (hx, cx)


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
    forcing_data = pd.read_csv(forcing_file_path, sep='\s+', names=forcing_columns_name, header=None,
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

        # data = torch.stack(out_SH2O)
        # optimizer.zero_grad()
        # loss = F.mse_loss(data, torch.rand(data.size()))
        # loss.backward()
        # plot_loss()
        # for item in grads:
        #     if torch.isnan(grads[item]).any():
        #         print(item)
        #         print(grads[item])
        # if torch.isnan(grad_soil_parameter.soilParam_withGrad.grad).any():
        #     print("nan gradient detected")

        # if k_time >= 1:
        #     dot = make_dot(SH2O, params=dict(grad_soil_parameter.named_parameters()), show_attrs=True, show_saved=False)
        #     dot.render('graph-output/forward-graph', format='pdf', view=True)
        #     optimizer.zero_grad()
        #     data = torch.stack(out_SH2O)
        #     loss = F.mse_loss(data, torch.rand(data.size()))
        #     loss.backward(retain_graph=True)
        #     dot = make_dot(loss, params=dict(grad_soil_parameter.named_parameters()), show_attrs=False, show_saved=True)
        #     dot.render('graph-output/backward_graph', format='pdf', view=True)
        #     break

    # # dot.graph_attr.update(size="500,500!")
    # dot.node_attr.update({
    #     # 'shape': 'ellipse',
    #     'shape': 'box',
    #     'style': 'filled',
    #     'fillcolor': 'lightgreen',
    #     'color': 'red',
    #     'fontcolor': 'blue',
    #     'fontsize': '12',
    #     'fontname': 'Helvetica',
    #     'height': '0.5',
    #     'width': '0.5',
    #     'label': 'Custom Node'
    # })
    # dot.graph_attr.update(size="100,100!")  # 调整图像大小
    # dot.graph_attr.update(dpi="600")  # 增加图像分辨率

    """output"""
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
                     index=Date[:k_time]).to_csv(os.path.join(output_dir, "NoahPy_output.txt"), index=True, sep=' ')
    condition = (Date >= startdate) & (Date <= enddate)
    return Date[condition], torch.stack(out_STC), torch.stack(out_SH2O)


def call_noah():
    """根据扰动的参数数据集运行Noah"""
    Noah_working_directory = "D:\PyCharm Project\\NoahPy\\Noah\\Noah_341_MODIFIED"
    Noah_exe_path = "D:\PyCharm Project\\NoahPy\\Noah\\Noah_341_MODIFIED\\driver.exe"
    # soil_param_path = "./SOILPARM.TBL"
    args = ['D:\PyCharm Project\\NoahPy\\Noah\data\\forcing\TGL_Noah_Forcing.txt', 'Noah_output.txt']
    subprocess.run([Noah_exe_path] + args, cwd=Noah_working_directory)


def interpolation(origin_position, target_position, data):
    interpolated_values = []
    for xi in target_position:
        # 找到 xi 所在区间的左右端点索引
        idx = torch.searchsorted(origin_position, xi) - 1
        x0, x1 = origin_position[idx], origin_position[idx + 1]
        y0, y1 = data[:, idx], data[:, idx + 1]
        interpolated_value = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
        interpolated_values.append(interpolated_value)
    return torch.stack(interpolated_values, dim=1)


def plot_grad():
    SH2O_New_mean = list()
    SH2OOUT_mean = list()
    P_mean = list()
    RHSTT_mean = list()
    CI_mean = list()
    BI_mean = list()
    AI_mean = list()
    for key, value in grads.items():
        if key.__contains__('SH2O_New'):
            print(f"{key}: {value}")
            SH2O_New_mean.insert(0, torch.mean(value))
        if key.__contains__('SH2OOUT'):
            print(f"{key}: {value}")
            SH2OOUT_mean.insert(0, torch.mean(value))
        if key.__contains__('RHSTT'):
            print(f"{key}: {value}")
            RHSTT_mean.insert(0, torch.mean(value))
        if key.__contains__('P'):
            print(f"{key}: {value}")
            P_mean.insert(0, torch.mean(value))
        if key.__contains__('CI'):
            print(f"{key}: {value}")
            CI_mean.insert(0, torch.mean(value))
        if key.__contains__('AI'):
            print(f"{key}: {value}")
            AI_mean.insert(0, torch.mean(value))
        if key.__contains__('BI'):
            print(f"{key}: {value}")
            BI_mean.insert(0, torch.mean(value))
    plt.plot(SH2O_New_mean, label='SH2O_New_mean')
    plt.plot(SH2OOUT_mean, label='SH2OOUT_mean')
    plt.plot(RHSTT_mean, label='RHSTT_mean')
    plt.plot(P_mean, label='P_mean')
    plt.plot(CI_mean, label='CI_mean')
    plt.plot(AI_mean, label='AI_mean')
    plt.plot(BI_mean, label='BI_mean')
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()


def train():
    observation_start_time = pd.to_datetime('2007-4-1')
    observation_end_time = pd.to_datetime('2010-12-31')
    """load 观测数据"""
    TGL_VWC_GT = "D:\\桌面\\Noah_Test\\TGL\\TGL_VWC_GT.csv"
    TGL_VWC_GT_columns = ['5cmVWC', '10cmVWC', '40cmVWC', '105cmVWC', '140cmVWC', '245cmVWC']
    TGL_VWC_GT = pd.read_csv(TGL_VWC_GT, header=0, index_col=0)
    TGL_VWC_GT = TGL_VWC_GT[TGL_VWC_GT_columns]
    TGL_VWC_GT.index = pd.to_datetime(TGL_VWC_GT.index)
    observation_condition = (TGL_VWC_GT.index >= observation_start_time) & (
            TGL_VWC_GT.index <= observation_end_time)
    TGL_VWC_GT = TGL_VWC_GT[observation_condition]
    Date = TGL_VWC_GT.index
    TGL_VWC_GT = tensor(TGL_VWC_GT.to_numpy(dtype=float), dtype=torch.float32)
    soil_depth = tensor(
        [0.045, 0.091, 0.166, 0.289, 0.493, 0.829, 1.383, 2.296, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 11.2, 13.2,
         15.2])

    midpoint_each_soil_layer = tensor(
        [0.022, 0.068, 0.1285, 0.2275, 0.391, 0.661, 1.014, 1.45, 1.95, 2.45, 3.15, 4.1, 5.1, 6.6, 8.6, 10.6, 12.6,
         14.6])

    observation_depth = tensor([0.05, 0.1, 0.4, 1.05, 1.4, 2.45])
    forcing_file = "../data/forcing/TGL_Noah_Forcing.txt"

    criterion = NSELoss()
    clip_values = {
        'BB': 0.1,
        'MAXSMC': 0.1,
        'SATDK': 1E-5,
        'SATPSI': 0.1,
        'QTZ': 0.1
    }
    lstm_input_size = 10
    lstm_hidden_size = 128
    lstm_output_size = 3
    lstm_model = LSTM(lstm_input_size, lstm_hidden_size, lstm_output_size)
    mod_dir = "./mod"
    os.makedirs(mod_dir, exist_ok=True)
    best_val_loss = np.inf
    MSE_record = []
    NSE_record = []
    optimizer_adam = torch.optim.Adam([
        {'params': Soil_Parameter.BB, 'lr': 0.01, 'momentum': 0.9},
        {'params': Soil_Parameter.MAXSMC, 'lr': 0.01, 'momentum': 0.9},
        {'params': Soil_Parameter.SATDK, 'lr': 0.01, 'momentum': 0.9},
        {'params': Soil_Parameter.SATPSI, 'lr': 0.01, 'momentum': 0.9},
        {'params': Soil_Parameter.QTZ, 'lr': 0.01, 'momentum': 0.9}
    ])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_sgd, step_size=10, gamma=0.5)

    switch_epoch = 30
    optimizer_sgd = None
    scheduler = None
    best_model_state = None
    for epoch in range(300):
        if epoch == switch_epoch:
            print("optimizer modification")
            Soil_Parameter.load_state_dict(best_model_state)
            optimizer_sgd = torch.optim.SGD([
                {'params': Soil_Parameter.BB, 'lr': 0.01, 'momentum': 0.9},
                {'params': Soil_Parameter.MAXSMC, 'lr': 0.01, 'momentum': 0.9},
                {'params': Soil_Parameter.SATDK, 'lr': 0.01, 'momentum': 0.9},
                {'params': Soil_Parameter.SATPSI, 'lr': 0.01, 'momentum': 0.9},
                {'params': Soil_Parameter.QTZ, 'lr': 0.01, 'momentum': 0.9}
            ])
            # 动态调整学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_sgd, mode='min', factor=0.1, patience=10,
                                                                   min_lr=1e-6)
        if epoch < switch_epoch:
            optimizer_adam.zero_grad()
        else:
            optimizer_sgd.zero_grad()
        Date, STC, SH2O = noah_main(forcing_file)
        data = interpolation(midpoint_each_soil_layer, observation_depth, SH2O)
        loss = criterion(data, TGL_VWC_GT)
        print(f"Epoch: {epoch + 1}  Loss: {loss.item()}")
        loss.backward()

        # plot_loss()

        NSE = calculate_nse(data, TGL_VWC_GT)
        NSE_record.append(NSE.item())
        MSE_record.append(utils.calculate_mse(data, TGL_VWC_GT).item())
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            state_dict = Soil_Parameter.state_dict()
            best_model_state = {key: value.clone() for key, value in state_dict.items()}
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
            torch.save(best_model_state,
                       '{}/{}_{:.4f}_{:.4f}.pth'.format(mod_dir, formatted_time, NSE, loss))
            print("Save in:",
                  '{}/{}_{:.4f}_{:.4f}.pth'.format(mod_dir, formatted_time, NSE, loss))
        # plot_timeseries(Date, data.detach().numpy(), TGL_VWC_GT.detach().numpy(), 'Unfrozen water content', '$UWC(m^3/m^3$)')
        Soil_Parameter.clamp_grad(clip_values)
        if epoch < switch_epoch:
            optimizer_adam.step()
        else:
            optimizer_sgd.step()
            scheduler.step(loss)
        Soil_Parameter.clamp()
    NSE_record = np.array(NSE_record)
    MSE_record = np.array(MSE_record)
    np.save('loss_record/NSE_record.npy', NSE_record)
    np.save('loss_record/MSE_record.npy', MSE_record)
    plot_loss(MSE_record, NSE_record)


def plot_loss(MSE_record, NSE_record):
    min_mse_record = []
    max_nse_record = []
    current_Min_MSE = float('inf')
    current_Max_NSE = -float('inf')

    for mse, nse in zip(MSE_record, NSE_record):
        if mse < current_Min_MSE:
            current_Min_MSE = mse
        if nse > current_Max_NSE:
            current_Max_NSE = nse
        min_mse_record.append(current_Min_MSE)
        max_nse_record.append(current_Max_NSE)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    # 绘制损失值曲线
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE', color=color)
    ax1.plot(min_mse_record, linestyle='--', marker=None, color=color, label='MSE Path', markersize=5)
    ax1.scatter(range(len(MSE_record)), MSE_record, color=color, label='MSE Points', s=8)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个 y 轴，共享 x 轴
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('NSE', color=color)
    ax2.plot(max_nse_record, linestyle='--', marker=None, color=color, label='NSE Path', markersize=5)
    ax2.scatter(range(len(NSE_record)), NSE_record, color=color, label='NSE Points', s=8)
    ax2.tick_params(axis='y', labelcolor=color)
    # 添加图例
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('MSE and NSE Curves')
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.savefig('{}/{}_NoahPy_Loss.png'.format('./loss_record/', formatted_time), dpi=600)
    plt.show()


def plot_noah_simulate():
    observation_start_time = pd.to_datetime('2007-4-1')
    observation_end_time = pd.to_datetime('2010-12-31')
    """load 观测数据"""
    TGL_VWC_GT = "E:\\Noah_LSM\\TGL\\\TGL_VWC_GT.csv"
    TGL_VWC_GT_columns = ['5cmVWC', '10cmVWC', '40cmVWC', '105cmVWC', '245cmVWC']
    TGL_VWC_GT = pd.read_csv(TGL_VWC_GT, header=0, index_col=0)
    TGL_VWC_GT = TGL_VWC_GT[TGL_VWC_GT_columns]
    TGL_VWC_GT.index = pd.to_datetime(TGL_VWC_GT.index)
    observation_condition = (TGL_VWC_GT.index >= observation_start_time) & (
            TGL_VWC_GT.index <= observation_end_time)
    TGL_VWC_GT = TGL_VWC_GT[observation_condition]
    Date = TGL_VWC_GT.index
    TGL_VWC_GT = tensor(TGL_VWC_GT.to_numpy(dtype=float), dtype=torch.float32)
    soil_depth = tensor(
        [0.045, 0.091, 0.166, 0.289, 0.493, 0.829, 1.383, 2.296, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 11.2, 13.2,
         15.2])

    midpoint_each_soil_layer = tensor(
        [0.022, 0.068, 0.1285, 0.2275, 0.391, 0.661, 1.106, 1.8395, 2.748, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 10.2, 12.2,
         14.2])
    # midpoint_each_soil_layer = tensor(
    #     [0.022, 0.068, 0.1285, 0.2275, 0.391, 0.661, 1.014, 1.45, 1.95, 2.45, 3.15, 4.1, 5.1, 6.6, 8.6, 10.6, 12.6,
    #      14.6])
    observation_depth = tensor([0.05, 0.1, 0.4, 1.05, 2.45])
    mod_dir = "D:\\PyCharm Project\\NoahPy\\Noah_341_MODIFIED_TORCH\mod"
    file_name = "D:\\PyCharm Project\\NoahPy\\Noah\\data\\forcing\\Origin_TGL_Noah_Forcing.txt"
    # state_dict = torch.load('{}/{}'.format(mod_dir, "2024-07-08-14-41-30_0.7773_0.2227.pth"))
    # Soil_Parameter.load_state_dict(state_dict)
    Noah_Date, STC, SH2O = noah_main(file_name)

    data = interpolation(midpoint_each_soil_layer, observation_depth, SH2O).detach().numpy()
    observation_condition = (Noah_Date >= observation_start_time) & (Noah_Date <= observation_end_time)
    data = data[observation_condition]
    plot_timeseries(Date, ["Observation", "NoahPy"], data, TGL_VWC_GT.detach().numpy(),
                    'Unfrozen water content',
                    '$UWC(m^3/m^3$)', [0.05, 0.1, 0.4, 1.05, 2.45], y_lim=0.4)


def plot_noah_noahpy():
    Noah_output = "D:\PyCharm Project\\NoahPy\\Noah\data\simulate\\Noah_output.csv"
    NoahPy_output = "D:\PyCharm Project\\NoahPy\\Noah\data\simulate\\NoahPy_output.csv"
    Noah_output = pd.read_csv(Noah_output, header=0, index_col=0, skipinitialspace=True)
    SM_columns = ['SH2O(1)', 'SH2O(2)', 'SH2O(3)', 'SH2O(4)', 'SH2O(5)', 'SH2O(6)', 'SH2O(7)', 'SH2O(8)', 'SH2O(9)']
    ST_columns = ['STC(1)', 'STC(2)', 'STC(3)', 'STC(4)', 'STC(5)', 'STC(6)', 'STC(7)', 'STC(8)', 'STC(9)']
    NoahPy_output = pd.read_csv(NoahPy_output, header=0, index_col=0)
    Date = pd.to_datetime(NoahPy_output.index)
    Noah_output.columns = Noah_output.columns.str.replace(' ', '')  # 去掉空格
    # SM_columns = [item for item in Noah_output.columns if item.__contains__('SH2O')]
    # ST_columns = [item for item in Noah_output.columns if item.__contains__('ST')]
    depths = [0.05, 0.1, 0.15, 0.3, 0.5, 0.8, 1.4, 2.3, 3.2]
    plot_timeseries(Date, ["Noah", "NoahPy"], Noah_output[SM_columns].to_numpy(), NoahPy_output[SM_columns].to_numpy(),
                    'Unfrozen water content',
                    '$UWC(m^3/m^3$)', depths, y_lim=0.3)
    plot_timeseries(Date, ["Noah", "NoahPy"], Noah_output[ST_columns].to_numpy(), NoahPy_output[ST_columns].to_numpy(),
                    'Unfrozen water content',
                    '$UWC(m^3/m^3$)', depths)


if __name__ == "__main__":
    # MSE_record = np.load('./loss_record/MSE_record.npy')
    # NSE_record = np.load('./loss_record/NSE_record.npy')
    # plot_loss(MSE_record, NSE_record)
    # plot_noah_simulate()
    # current_Min_MSE = [0.8, 0.6, 0.7, 0.5, 0.3, 0.4, 0.2]
    # current_Max_NSE = [0.1, 0.2, 0.15, 0.25, 0.3, 0.28, 0.35]
    # plot_loss(current_Min_MSE, current_Max_NSE)
    # plot_noah_simulate()
    # train()
    # file_name = "H:\\Noah_LSM\Observation\site_all_process\grid_average_check\\27.75_89.15\\27.75_89.15_plant.txt"
    # file_name = "H:\\Noah_LSM\Observation\site_all_process\grid_average_check\\37.85_101.15\\37.85_101.15_plant.txt"
    file_name = "H:\\Noah_LSM\\Observation\site_all_process\grid_site_scale\\use\sourth\QOMS\\forcing_QOMS_plant.txt"
    profiler = cProfile.Profile()
    profiler.enable()  # 开始性能分析
    # lstm_input_size = 10
    # lstm_hidden_size = 128
    # lstm_output_size = 3
    # lstm_model = LSTM(lstm_input_size, lstm_hidden_size, lstm_output_size)
    Date, STC, SH2O = noah_main(file_name, output_flag=False, lstm_model=None)

    profiler.disable()  # 停止性能分析
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')  # 按累计时间排序
    stats.print_stats()
    print(stream.getvalue())  # 打印性能分析结果
    # call_noah()0
    # noah_main(file_name, output_flag=True)
