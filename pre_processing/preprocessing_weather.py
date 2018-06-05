import os
import STRING
import pandas as pd
import utils.dataframe_utils as dfutils
import datetime
import utils.model_utils as mdutils

os.chdir(STRING.PATH.ROOT_DIR)

def preprocessing(file):
    df = pd.read_csv(file, sep=',', encoding='utf-8')

    # Rename the variables
    dict_rename = {'Fecha':'DATE', 'Precipitación como día pluviométrico(mm)': 'PRECIPITATION_7_7(mm)',
                   'Temperatura Máxima del aire en abrigo a 1,5 m de altura(ºC)': 'TEMP_MAX_1.5m',
                   'Temperatura Mínima del aire  en abrigo a 1,5 m de altura(ºC)': 'TEMP_MIN_1.5m',
                   'Humedad relativa media(%)': 'HUMIDITY(%)', 'Humedad relativa 8-14-20(%)': 'HUMIDITY_8_14_20(%)',
                   'Velocidad media del viento a 2m(km/h)': 'WIND_SPEED_2m(km/h)',
                   'Dirección prevalente del viento a 2m(PC)': 'WIND_PREV_2m(PC)',
                   'Dirección prevalente del viento a 10m altura(PC)': 'WIND_DIRECTION_10m(PC)',
                   'Duración del follaje mojado(h)': 'WET_FOLIAGE(h)',
                   'Velocidad media del viento a 10m(km/h)': 'WIND_SPEED_10m(km/h)',
                   'Tensión de vapor media(hPa)': 'STEAM_TENSION (hPa)',
                   'Evapotranspiración Potencial(mm)': 'EVAPOTRANSPIRATION (mm)',
                   'Unidades de Frío(h)': 'COLD_UNITIES (h)',
                   'Temperatura Media del aire en abrigo a 1,5 m de altura(ºC)': 'TEMP_1.5m',
                   'Evaporación del tanque(mm)': 'EVAPORATION (mm)',
                   'Temperatura mínima del aire en intemperie a 0,5m(ºC)': 'TEMP_MIN_OUTSIDE_0.5m',
                   'Heliofanía Efectiva(h)': 'SUN_BRIGHT(h)', 'Radiación Neta(MJ/m2)': 'RADIATION (MJ/m2)',
                   'Punto de rocío medio(ºC)': 'DEW (ºC)', 'Máxima velocidad del viento(km/h)': 'WIND_MAX_SPEED(km/h)',
                   'Máxima precipitación en 30min(mm)': 'PRECIPITATION_MAX_30min (mm)',
                   'Horas de Frío(h)': 'COLD_HOURS (h)', 'Presión atmosférica media(hPa)': 'ATM_PRESSURE (hPa)',
                   'Temperatura media del suelo a 10cm de profundidad(ºC)': 'TEMP_GROUND_10cm',
                   'Temperatura mínima del aire en intemperie a 1.5m(ºC)': 'TEMP_MIN_OUTSIDE_1.5m',
                   'Heliofanía Relativa(%)': 'RELATIVE_SUN_BRIGHT(%)'}

    df = df.rename(columns=dict_rename)

    # CHANGE VAR TYPE
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')
    for i in df.columns.values.tolist():
        if i != 'DATE':
            df[i] = pd.to_numeric(df[i], errors='coerce')

    # DROP NAN VALUES & MULTI-OUTPUT FILL
    df = df.dropna(how='all', axis=1)
    cols = df.columns.values.tolist()
    cols.remove('DATE')
    df = df.dropna(subset=cols, how='all', axis=0)

    df = mdutils.fillna_multioutput(df, not_consider=['DATE'], on='DATE')

    # STATISTICS
    dfutils.statistic_df(df)
    df.to_csv(STRING.PATH.file_preprocessed_clima, sep=';', encoding='latin1', index=False)


if __name__ == '__main__':
    preprocessing(STRING.PATH.path_weather + STRING.PATH.file_weather_raw)
