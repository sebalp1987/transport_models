import pandas as pd
import STRING
import utils.process_utils as procc_utils
import os

os.chdir(STRING.PATH.ROOT_DIR)

clima = pd.read_csv(STRING.PATH.file_preprocessed_clima, encoding='latin1', delimiter=';')
tren = pd.read_csv(STRING.PATH.file_preprocessed_tren, encoding='latin1', delimiter=';')
subte = pd.read_csv(STRING.PATH.file_preprocessed_subte, encoding='latin1', delimiter=';')
bus = pd.read_csv(STRING.PATH.file_preprocessed_bus_day, encoding='latin1', delimiter=';')
bus['DATE'] = pd.to_datetime(bus['DATE'])
tren['DATE'] = pd.to_datetime(tren['DATE'])
subte['DATE'] = pd.to_datetime(subte['DATE'])
clima['DATE'] = pd.to_datetime(clima['DATE'])
bus = bus.fillna(0)
keys = ['DATE']

# DUMMY GOVERN
bus['DUMMY_GOVERN'] = pd.Series(0, index=bus.index)
bus.loc[bus['DATE'] >= '01-11-2015', 'DUMMY_GOVERN'] = 1
bus = bus.drop_duplicates(subset=['DATE'], keep='first')

# MERGE DATAFRAME
# Train
'''
tren = tren[['DATE', 'PASSENGER_SUM_DAY', 'INCOME_SUM_DAY', 'PASSENGER_MAX_DAY', 'PASSENGER_MIN_DAY', 'PASSENGER_MEDIAN_DAY',
            'PASSENGER_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MIN_DAY', 'INCOME_MEDIAN_DAY', 'INCOME_AVG_DAY', 'FARE_AVG']]
'''
tren = tren[['DATE', 'PASSENGER_SUM_DAY', 'FARE_AVG']]

tren = tren.add_suffix('_TRAIN')
tren['DATE'] = tren['DATE_TRAIN']
del tren['DATE_TRAIN']

# Metro
'''
subte = subte[['DATE', 'PASSENGER_SUM_DAY', 'INCOME_SUM_DAY', 'PASSENGER_MAX_DAY', 'PASSENGER_MIN_DAY', 'PASSENGER_MEDIAN_DAY',
               'PASSENGER_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MIN_DAY', 'INCOME_MEDIAN_DAY', 'INCOME_AVG_DAY',
               'FARE_AVG']]
'''
subte = subte[['DATE', 'PASSENGER_SUM_DAY', 'FARE_AVG']]

subte = subte.add_suffix('_METRO')
subte['DATE'] = subte['DATE_METRO']
del subte['DATE_METRO']

date = bus[['DATE']]
date = date.drop_duplicates()
clima = pd.merge(date, clima, how='left', on='DATE')
clima = clima.sort_values(by=['DATE'], ascending=[True])
clima = clima.interpolate(limit_direction='backward', method='nearest')
bus = pd.merge(bus, tren, how='left', on='DATE')
bus = pd.merge(bus, subte, how='left', on='DATE')
bus = pd.merge(bus, clima, how='left', on='DATE')

bus['DATE'] = pd.to_datetime(bus['DATE'])

# ECONOMIC INDICATORS: PBI, UNEMPLOYMENT, DOLAR VARIATION, POVERTY, SALARIO, INFLATION RATES, ESTMADOR MENSUAL
# INDUSTRIAL, ESTIMADOR MENSUAL ACTIVIDAD ECONOMICA

bus['YEAR'] = pd.DatetimeIndex(bus['DATE']).year
bus['MONTH'] = pd.DatetimeIndex(bus['DATE']).month

dolar = pd.read_csv('files\\economy_file\\dolar_per_day.csv', delimiter=';', encoding='latin1')
dolar['DATE'] = pd.to_datetime(dolar['DATE'], format='%d/%m/%Y', errors='coerce')
wages = pd.read_csv('files\\economy_file\\wages.csv', delimiter=';', encoding='latin1')
wages['DATE'] = pd.to_datetime(wages['DATE'], format='%d/%m/%Y', errors='coerce')
inflation = pd.read_csv('files\\economy_file\\inflation.csv', delimiter=';', encoding='latin1')

dolar = pd.merge(date, dolar, how='left', on='DATE')
dolar = dolar.sort_values(by=['DATE'], ascending=[True])
dolar = dolar.interpolate(limit_direction='backward', method='nearest')


bus = pd.merge(bus, dolar, how='left', on='DATE')
bus = pd.merge(bus, wages, how='left', on='DATE')
bus = pd.merge(bus, inflation, how='left', on=['MONTH', 'YEAR'])
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

print(bus.isnull().sum())
bus = bus.dropna(axis=0, how='any')
del_var = ['PASSENGERS', 'INCOME']
for i in del_var:
    del bus[i]

bus['DLS_LAST'] = bus['DLS_LAST'].map(float)
bus['WAGE'] = bus['WAGE'].map(float)
bus['PRICE_LEVEL'] = bus['PRICE_LEVEL'].map(float)

for i in bus.columns.values.tolist():
    if 'FARE_AVG' in i or 'INCOME' in i:
        var_name_dls = i + '_DLS'
        var_name_wage = i + '_WAGE'
        var_name_inflation = i + '_IPC'
        bus[i] = bus[i].map(float)
        bus[var_name_dls] = bus[i] / bus['DLS_LAST']
        bus[var_name_wage] = bus[i] / bus['WAGE']
        bus[var_name_inflation] = bus[i] / bus['PRICE_LEVEL']

del dolar, wages, inflation

# FARE VARIATONS-RELATIVES-ETC
bus['FARE_BUS_TRAIN'] = bus['FARE_AVG'] / bus['FARE_AVG_TRAIN']
bus['FARE_BUS_METRO'] = bus['FARE_AVG'] / bus['FARE_AVG_METRO']
bus['FARE_METRO_TRAIN'] = bus['FARE_AVG_METRO'] / bus['FARE_AVG_TRAIN']

bus['PASSENGERS_BUS_TRAIN'] = bus['PASSENGER_SUM_DAY'] / bus['PASSENGER_SUM_DAY_TRAIN']
bus['PASSENGERS_BUS_METRO'] = bus['PASSENGER_SUM_DAY'] / bus['PASSENGER_SUM_DAY_METRO']
bus['PASSENGERS_METRO_TRAIN'] = bus['PASSENGER_SUM_DAY'] / bus['PASSENGER_SUM_DAY_TRAIN']

del bus['mes']
# LAGS
'''
shifted_var = ['FARE_AVG', 'INCOME_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MIN_DAY', 'INCOME_SUM_DAY',
                              'INTERN_AVG_HOUR', 'INTERN_COUNT_DAY', 'INTERN_MAX_HOUR', 'INTERN_MEDIAN_HOUR',
                              'INTERN_MIN_HOUR', 'DAY_holiday', 'PASSENGER_AVG_DAY', 'PASSENGER_MAX_DAY',
                              'PASSENGER_MEDIAN_DAY', 'PASSENGER_MIN_DAY', 'PASSENGER_SUM_DAY', 'WORKDAY',
                              'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2', 'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5',
                              'WEEKDAY_6', 'PASSENGER_SUM_DAY_TRAIN', 'INCOME_SUM_DAY_TRAIN',
                              'PASSENGER_MAX_DAY_TRAIN', 'PASSENGER_MIN_DAY_TRAIN', 'PASSENGER_MEDIAN_DAY_TRAIN',
                              'PASSENGER_AVG_DAY_TRAIN', 'INCOME_MAX_DAY_TRAIN', 'INCOME_MIN_DAY_TRAIN',
                              'INCOME_MEDIAN_DAY_TRAIN', 'INCOME_AVG_DAY_TRAIN', 'FARE_AVG_TRAIN',
                              'PASSENGER_SUM_DAY_METRO', 'INCOME_SUM_DAY_METRO', 'PASSENGER_MAX_DAY_METRO',
                              'PASSENGER_MIN_DAY_METRO', 'PASSENGER_MEDIAN_DAY_METRO', 'PASSENGER_AVG_DAY_METRO',
                              'INCOME_MAX_DAY_METRO', 'INCOME_MIN_DAY_METRO', 'INCOME_MEDIAN_DAY_METRO',
                              'INCOME_AVG_DAY_METRO', 'FARE_AVG_METRO', 'PRECIPITACION-MM', 'PRECIPITATION_7_7(mm)',
                              'TEMP_MAX_1.5m', 'TEMP_MIN_1.5m', 'HUMIDITY(%)', 'HUMIDITY_8_14_20(%)',
                              'WIND_SPEED_2m(km/h)', 'WET_FOLIAGE(h)', 'WIND_SPEED_10m(km/h)',
                              'STEAM_TENSION (hPa)', 'COLD_UNITIES (h)', 'TEMP_1.5m', 'TEMP_MIN_OUTSIDE_0.5m',
                              'DEW (ºC)', 'WIND_MAX_SPEED(km/h)', 'PRECIPITATION_MAX_30min (mm)', 'COLD_HOURS (h)',
                              'ATM_PRESSURE (hPa)', 'TEMP_GROUND_10cm', 'TEMP_MIN_OUTSIDE_1.5m',
                              'DLS_LAST', 'DLS_OPEN', 'DLS_MAX', 'DLS_MIN', 'DLS_VAR', 'WAGE', 'WAGE_VAR'
                              ]
'''

# VARIANCE THRESHOLD
bus = bus.reset_index(drop=True)
bus = procc_utils.variance_threshold(bus, keys, threshold=0.02)

# OUTLIERS
'''
for i in bus.columns.values.tolist():
    if i not in keys:
        Outliers.outliers_mad(bus, i, not_count_zero=False, just_count_zero=True)
'''
print(bus['WIND_SPEED_2m(km/h)'])

bus['WIND_STRONG'] = pd.Series(0, index=bus.index)
bus.loc[bus['WIND_SPEED_2m(km/h)'] > 8, 'WIND_STRONG'] = 1


# Robust
# bus = procc_utils.robust_scale(bus, key_vars=keys)

# PCA REDUCTION
# bus = procc_utils.pca_reduction(bus, normalization=False)
date = date.sort_values(by=['DATE'], ascending=[True])
date = date.reset_index(drop=True)

date['TREND'] = pd.Series(date.index, index=date.index)
bus = pd.merge(bus, date, how='left', on='DATE')

for i in bus.columns.values.tolist():
    if i.startswith('DAY_') or i.startswith('YEAR_'):
        del bus[i]

cols = bus.columns.values.tolist()
bus = bus[['PASSENGER_SUM_DAY'] + [x for x in cols if not 'PASSENGER_SUM_DAY' in x]]
del bus['MONTH']

bus.to_csv('temporal_data.csv', sep=';', encoding='latin1', index=False)