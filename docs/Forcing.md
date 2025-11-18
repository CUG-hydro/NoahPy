# NoahPy Forcing Data Documentation

## Overview

The forcing.txt file provides atmospheric forcing data and model configuration parameters required to run the Noah Land Surface Model in NoahPy. The file consists of two main sections:

1. **Metadata Section**: Model configuration and initial conditions
2. **Forcing Data Section**: Time-series atmospheric forcing variables

---

## File Structure

```
&METADATA_NAMELIST
[metadata parameters]
/
--------------------------------------------------------------------------------------------
[column headers]
[units]
--------------------------------------------------------------------------------------------
<Forcing>
[time-series forcing data]
```

---

## Metadata Variables

### Simulation Period

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `startdate` | - | Simulation start date and time in format "YYYYMMDDHHMM" | Specify as string, e.g., "200704010000" for April 1, 2007 00:00 |
| `enddate` | - | Simulation end date and time in format "YYYYMMDDHHMM" | Specify as string, e.g., "201012310000" for December 31, 2010 00:00 |
| `loop_for_a_while` | - | Number of iterations to loop (0 = run once) | Set to 0 for single simulation run |

### Location Parameters

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Latitude` | degrees | Latitude of the simulation point | Decimal degrees, positive for North, negative for South (e.g., 33.02) |
| `Longitude` | degrees | Longitude of the simulation point | Decimal degrees, positive for East, negative for West (e.g., 91.96) |

### Time Step Configuration

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Forcing_Timestep` | seconds | Time step interval of forcing data | Typically 3600 (1 hour) or 86400 (1 day) |
| `Noahlsm_Timestep` | seconds | Noah LSM internal computation time step | Should be ≤ Forcing_Timestep, typically 86400 for daily data |

### Output Configuration

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `output_dir` | - | Directory path for output files | Leave empty ("") to output in current directory or specify full path |

### Surface Type

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Sea_ice_point` | - | Whether the point is sea ice | Set to `.FALSE.` for land points, `.TRUE.` for sea ice |

### Soil Layer Configuration

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Soil_layer_thickness` | m | Thickness of each soil layer (space-separated values) | Specify thickness for each layer, typically 18-20 layers. Example: `0.045 0.046 0.075 0.123 0.204...` Default layers from top to bottom |
| `Soil_htype` | - | Soil type index for each layer | Soil type classification (1-19), space-separated for each layer. See NOAH soil classification table |

### Initial Soil State

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Soil_Temperature` | K | Initial temperature for each soil layer | Space-separated values in Kelvin for each layer. Can be obtained from spin-up run or field measurements |
| `Soil_Moisture` | m³/m³ | Initial volumetric soil moisture for each layer | Volumetric water content (0-1) for each layer. Should be between wilting point and porosity |
| `Soil_Liquid` | m³/m³ | Initial liquid water content for each layer | Liquid water content (unfrozen) for each layer. Should be ≤ Soil_Moisture |

### Surface Initial Conditions

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Skin_Temperature` | K | Initial land surface skin temperature | Surface temperature in Kelvin, typically close to air temperature |
| `Canopy_water` | kg/m² | Initial canopy water content | Intercepted water on vegetation, typically 0-5 kg/m² |
| `Snow_depth` | m | Initial snow depth | Physical snow depth in meters, 0 for no snow |
| `Snow_equivalent` | m | Initial snow water equivalent | Water equivalent of snow pack in meters |
| `Deep_Soil_Temperature` | K | Bottom boundary soil temperature | Deep soil temperature (at lower boundary), typically annual mean temperature |

### Land Surface Classification

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Landuse_dataset` | - | Land use classification system | Specify "USGS" or "MODIS" |
| `Soil_type_index` | - | Dominant soil type for the point | Integer 1-19 based on NOAH soil classification |
| `Vegetation_type_index` | - | Vegetation type index | Integer based on USGS (1-27) or MODIS classification |
| `Urban_veg_category` | - | Urban vegetation category index | Typically 1 for urban areas in USGS classification |
| `glacial_veg_category` | - | Glacier/ice vegetation category | Typically 24 for glacier in USGS classification |
| `Slope_type_index` | - | Terrain slope category | Integer 1-9 representing slope class |

### Radiation Parameters

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Max_snow_albedo` | - | Maximum snow albedo (0-1) | Fraction (0.5-0.85), typically 0.75 for fresh snow |

### Atmospheric Measurement Levels

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Air_temperature_level` | m | Height of air temperature measurement | Height above ground in meters, typically 2.0 m |
| `Wind_level` | m | Height of wind measurement | Height above ground in meters, typically 10.0 m |

### Vegetation Parameters

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Green_Vegetation_Min` | - | Minimum green vegetation fraction (0-1) | Minimum seasonal vegetation cover, e.g., 0.01 |
| `Green_Vegetation_Max` | - | Maximum green vegetation fraction (0-1) | Maximum seasonal vegetation cover, e.g., 0.96 |

### Model Options

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Usemonalb` | - | Use monthly albedo climatology | `.FALSE.` to use model-computed albedo, `.TRUE.` for monthly values |
| `Rdlai2d` | - | Read 2D LAI from input | `.FALSE.` to use monthly climatology, `.TRUE.` to read from forcing data |
| `sfcdif_option` | - | Surface layer drag coefficient scheme | 0 = original Noah, 1 = MYJ scheme |
| `iz0tlnd` | - | Thermal roughness length option | 0 = use CZIL, other values for different formulations |

### Monthly Climatology Arrays

These parameters provide monthly values (12 values, one for each month):

| Variable | Unit | Description | How to Prepare |
|----------|------|-------------|----------------|
| `Albedo_monthly` | - | Monthly surface albedo (12 values) | Background albedo for each month (Jan-Dec), e.g., `0.18 0.17 0.16 0.15 0.15 0.15 0.15 0.16 0.16 0.17 0.17 0.18` |
| `Shdfac_monthly` | - | Monthly vegetation shading fraction (12 values) | Green vegetation fraction for each month (0-1) |
| `lai_monthly` | m²/m² | Monthly leaf area index (12 values) | LAI for each month, e.g., `4.0 4.0 4.0...` |
| `Z0brd_monthly` | m | Monthly background roughness length (12 values) | Roughness length for each month in meters |

---

## Forcing Data Columns

The forcing data section begins after the `/` delimiter and the `<Forcing>` tag. Each row represents one time step.

| Column | Variable | Unit | Description | Data Requirements |
|--------|----------|------|-------------|-------------------|
| 1 | Year | yyyy | Year | 4-digit year |
| 2 | Month | mm | Month | 1-12 |
| 3 | Day | dd | Day of month | 1-31 |
| 4 | Hour | hh | Hour | 0-23 |
| 5 | Minute | mi | Minute | 0-59 |
| 6 | windspeed | m/s | Wind speed at reference height | Non-negative value, typically 0-30 m/s |
| 7 | winddir | degrees | Wind direction (meteorological convention) | 0-360°, where 0° = from North |
| 8 | temperature | K | Air temperature at reference height | In Kelvin, typically 220-320 K |
| 9 | humidity | % | Relative humidity | 0-100% |
| 10 | pressure | hPa | Surface atmospheric pressure | Typically 500-1000 hPa depending on elevation |
| 11 | shortwave | W/m² | Incoming shortwave radiation | Non-negative, 0 at night, typically 0-1200 W/m² |
| 12 | longwave | W/m² | Incoming longwave radiation | Typically 150-450 W/m² |
| 13 | precipitation | kg/m²/s | Precipitation rate | Non-negative, typically 0-0.01 kg/m²/s |
| 14 | LAI | m²/m² | Leaf Area Index (if `Rdlai2d=.TRUE.`) | Optional, typically 0-10 m²/m² |
| 15 | NDVI | - | Normalized Difference Vegetation Index | Optional, typically 0-1 |

---

## How to Prepare Forcing Data

### 1. Metadata Section

1. **Define simulation period**: Set `startdate` and `enddate` according to your study period
2. **Set location**: Specify `Latitude` and `Longitude` of your site
3. **Configure time steps**: Set `Forcing_Timestep` and `Noahlsm_Timestep` (must match your data temporal resolution)
4. **Set soil layers**: Define `Soil_layer_thickness` and corresponding `Soil_htype` for each layer
5. **Initialize states**: Set initial conditions for soil temperature, moisture, and surface variables
   - Can be obtained from spin-up runs
   - Or estimated from field measurements/reanalysis data
6. **Classify land surface**: Set `Vegetation_type_index`, `Soil_type_index` based on land cover maps
7. **Configure monthly parameters**: Provide 12 monthly values for albedo, vegetation fraction, LAI, and roughness length

### 2. Forcing Data Section

1. **Time series preparation**:
   - Ensure continuous time series without gaps
   - Time step must match `Forcing_Timestep` in metadata
   - Convert all times to consistent timezone (UTC recommended)

2. **Data quality control**:
   - Remove outliers and unrealistic values
   - Fill missing data using appropriate methods (interpolation, gap-filling)
   - Ensure physical consistency (e.g., shortwave radiation = 0 at night)

3. **Unit conversion**:
   - Temperature: Convert to Kelvin (K = °C + 273.15)
   - Pressure: Convert to hPa (1 hPa = 100 Pa)
   - Radiation: Ensure W/m² units
   - Precipitation: Convert to kg/m²/s (1 mm/hour = 1/3600 kg/m²/s)
   - Wind direction: Meteorological convention (direction wind comes FROM)

4. **Data sources**:
   - Meteorological station observations
   - Reanalysis products (ERA5, MERRA-2, GLDAS)
   - Remote sensing (radiation, LAI, NDVI)
   - Satellite precipitation products (GPM, TRMM)

### 3. File Format

1. Use space-delimited format
2. Metadata section enclosed in `&METADATA_NAMELIST` and `/`
3. Include column headers and units
4. Mark forcing data start with `<Forcing>` tag
5. Ensure consistent decimal precision (6-10 significant digits recommended)

---

## Example Forcing Data Format

```
&METADATA_NAMELIST
startdate = "200704010000"
enddate = "201012310000"
loop_for_a_while = 0
output_dir = ""
Latitude = 33.02
Longitude = 91.96
Forcing_Timestep = 86400
Noahlsm_Timestep = 86400
Sea_ice_point = .FALSE.
Soil_layer_thickness = 0.045 0.046 0.075 0.123 0.204 0.336 0.371 0.3 0.5 0.5 0.7 1.0 1.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0
Soil_Temperature = 273.4 271.6 271.2 270.8 270.2 269.4 268.7 268.4 267.5 268.5 270.5 271.0 271.4 271.6 271.7 271.8 271.8 271.9 271.8 271.9
Soil_Moisture = 0.104 0.22 0.247 0.222 0.276 0.247 0.238 0.223 0.205 0.169 0.337 0.337 0.337 0.094 0.109 0.121 0.131 0.15 0.164 0.173
Soil_Liquid = 0.104 0.099 0.104 0.084 0.052 0.03 0.003 0.01 0.031 0.057 0.035 0.039 0.042 0.016 0.017 0.018 0.019 0.021 0.023 0.024
Soil_htype = 7 7 7 7 9 7 7 7 7 7 12 12 12 13 13 13 13 13 13 13
Skin_Temperature = 271.241
Canopy_water = 0
Snow_depth = 0
Snow_equivalent = 0
Deep_Soil_Temperature = 272.43
Landuse_dataset = "USGS"
Soil_type_index = 3
Vegetation_type_index = 9
Urban_veg_category = 1
glacial_veg_category = 24
Slope_type_index = 3
Max_snow_albedo = 0.75
Air_temperature_level = 2.0
Wind_level = 10.0
Green_Vegetation_Min = 0.01
Green_Vegetation_Max = 0.96
Usemonalb = .FALSE.
Rdlai2d = .FALSE.
sfcdif_option = 1
iz0tlnd = 0
Albedo_monthly = 0.18 0.17 0.16 0.15 0.15 0.15 0.15 0.16 0.16 0.17 0.17 0.18
Shdfac_monthly = 0.01 0.02 0.07 0.17 0.27 0.58 0.93 0.96 0.65 0.24 0.11 0.02
lai_monthly = 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0
Z0brd_monthly = 0.02 0.02 0.025 0.03 0.035 0.036 0.035 0.03 0.027 0.025 0.02 0.02
/
------------------------------------------------------------------------------------------------------------------------------------------------------------
 UTC date/time        windspeed       wind dir         temperature      humidity        pressure           shortwave      longwave          precipitation
yyyy mm dd hh mi       m s{-1}        degrees               K               %             hPa               W m{-2}        W m{-2}          kg m{-2} s{-1}
------------------------------------------------------------------------------------------------------------------------------------------------------------
<Forcing>
2007 04 01 00 00     4.1889973491   249.0000000000   271.2412439391    25.9113713626   548.2200000000   243.6875000000   228.8125000000 0.0000000003104409
2007 04 02 00 00     4.7967473780   249.0000000000   269.9199939687    16.4887785222   548.2200000000   340.3437500000   191.5000000000 0.0000003475326554
2007 04 03 00 00     5.4277474080   249.0000000000   269.2887439828    24.7007722523   548.2200000000   315.1562500000   199.6562500000 0.0000004343382090
...
```

---

## Common Data Sources

### Meteorological Variables
- **Ground observations**: National weather stations, flux towers
- **Reanalysis**: ERA5, MERRA-2, JRA-55, NCEP/NCAR
- **Regional models**: WRF, RegCM outputs

### Radiation
- **Satellite**: CERES, GOES, MODIS
- **Ground**: BSRN, surfrad networks
- **Reanalysis**: ERA5 surface radiation

### Precipitation
- **Gauges**: National precipitation networks
- **Radar**: NEXRAD, weather radar products
- **Satellite**: GPM, TRMM, PERSIANN
- **Merged products**: MSWEP, CHIRPS

### Vegetation Indices
- **MODIS**: MOD13 (NDVI), MOD15 (LAI/FPAR)
- **AVHRR**: GIMMS NDVI
- **VIIRS**: VNP13, VNP15
- **Landsat**: Surface reflectance products

---

## Notes

1. **Initial conditions**: For long simulations, perform a spin-up run (1-10 years) to equilibrate soil states
2. **Time zones**: Ensure consistency between forcing data times and `startdate`/`enddate`
3. **Data gaps**: NoahPy requires continuous forcing data. Gap-fill or interpolate missing values before running
4. **Soil layers**: Total number of layers is flexible but typically 18-20 layers. Ensure all soil arrays have same length
5. **Frozen soil**: When soil temperature < 273.15 K, set `Soil_Liquid` < `Soil_Moisture` to represent ice content
6. **Parameter consistency**: Ensure vegetation and soil parameters are consistent with actual land cover

---

## References

- Noah LSM Documentation: https://ral.ucar.edu/solutions/products/noah-land-surface-model-lsm
- USGS Land Use Classification: https://www.usgs.gov/centers/eros/science/national-land-cover-database
- Soil texture classification: https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/

---

## Contact

For questions about forcing data preparation, please refer to the NoahPy repository or contact the development team.
