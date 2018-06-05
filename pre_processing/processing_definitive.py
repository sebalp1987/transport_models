import pandas as pd
import STRING
import matplotlib.pyplot as plot
import seaborn as sns
import utils.plot_utils as plot_utils
import utils.process_utils as procc_utils
from sklearn.preprocessing import scale
from utils.outliers import Outliers
import os
from sklearn import preprocessing as pp

os.chdir(STRING.PATH.ROOT_DIR)

clima = pd.read_csv(STRING.PATH.file_preprocessed_clima, encoding='latin1', delimiter=';')
tren = pd.read_csv(STRING.PATH.path_files + STRING.PATH.file_preprocessed_tren, encoding='latin1', delimiter=';')
subte = pd.read_csv(STRING.PATH.path_files + STRING.PATH.file_preprocessed_subte, encoding='latin1', delimiter=';')
bus = pd.read_csv(STRING.PATH.path_files + STRING.PATH.file_preprocessed_bus, encoding='latin1', delimiter=';')
bus['DATE'] = pd.to_datetime(bus['DATE'])
tren['DATE'] = pd.to_datetime(tren['DATE'])
subte['DATE'] = pd.to_datetime(subte['DATE'])
clima['DATE'] = pd.to_datetime(clima['DATE'])
bus = bus.fillna(0)
keys = ['DATE', 'SIZE']

# VARIABLE THAT MEASURES THE SIZE OF THE COMPANY
bus_size = bus[['ID_LINE', 'INTERN_MAX_HOUR']]
bus_size = bus_size.sort_values(by=['ID_LINE', 'INTERN_MAX_HOUR'], ascending=[True, False])
bus_size = bus_size.drop_duplicates(subset=['ID_LINE'], keep='first')
bus_max = bus_size['INTERN_MAX_HOUR'].max()
bus_min = bus_size['INTERN_MAX_HOUR'].min()
print(bus_size['INTERN_MAX_HOUR'].describe(percentiles=[.10, .20, .30, .40, .50, .60, .70, .80, .90]))
# plot_utils.hist_utils(bus, 'INTERN_MAX_HOUR')
bus_size['SIZE'] = pd.Series('', index=bus_size.index)


# DUMMY GOVERN
bus['DUMMY_GOVERN'] = pd.Series(0, index=bus.index)
bus.loc[bus['DATE'] >= '01-11-2015', 'DUMMY_GOVERN'] = 1

# SIZE
print(bus_min)
print(bus_size['INTERN_MAX_HOUR'].quantile(0.4))
print(bus_size['INTERN_MAX_HOUR'].quantile(0.7))

bus_size.loc[bus_size['INTERN_MAX_HOUR'].between(bus_min, bus_size['INTERN_MAX_HOUR'].quantile(0.4), inclusive=True), 'SIZE'] = '3'
bus_size.loc[bus_size['INTERN_MAX_HOUR'].between(bus_size['INTERN_MAX_HOUR'].quantile(0.4) + 1,
                                                 bus_size['INTERN_MAX_HOUR'].quantile(0.7), inclusive=True), 'SIZE'] = '2'
bus_size.loc[bus_size['INTERN_MAX_HOUR'].between(bus_size['INTERN_MAX_HOUR'].quantile(0.7) + 1, bus_max, inclusive=True), 'SIZE'] = '1'
del bus_size['INTERN_MAX_HOUR']

# GROUP BUS BY SIZE
bus = pd.merge(bus, bus_size, how='left', on='ID_LINE')
del bus['ID_LINE']

group_variables = ['DATE', 'SIZE', 'FARE_AVG', 'INCOME', 'INCOME_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MEDIAN_DAY',
                   'INCOME_MIN_DAY', 'INCOME_SUM_DAY',	'INTERN_AVG_HOUR', 'INTERN_COUNT_DAY',	'INTERN_MAX_HOUR',	
                   'INTERN_MEDIAN_HOUR', 'INTERN_MIN_HOUR',	'PASSENGERS', 'PASSENGER_AVG_DAY', 'PASSENGER_MAX_DAY',	
                   'PASSENGER_MEDIAN_DAY', 'PASSENGER_MIN_DAY',	'PASSENGER_SUM_DAY']

group_bus = bus[group_variables]

sum_var = ['INCOME', 'INCOME_AVG_DAY', 'INCOME_MAX_DAY', 'INCOME_MEDIAN_DAY',
                   'INCOME_MIN_DAY', 'INCOME_SUM_DAY',	'INTERN_AVG_HOUR', 'INTERN_COUNT_DAY',	'INTERN_MAX_HOUR',
                   'INTERN_MEDIAN_HOUR', 'INTERN_MIN_HOUR',	'PASSENGERS', 'PASSENGER_AVG_DAY', 'PASSENGER_MAX_DAY',
                   'PASSENGER_MEDIAN_DAY', 'PASSENGER_MIN_DAY',	'PASSENGER_SUM_DAY']

avg_var = ['FARE_AVG']

for i in sum_var:
    group_bus[i] = group_bus.groupby(by=['DATE', 'SIZE'])[i].transform('sum')

for i in avg_var:
    group_bus[i] = group_bus.groupby(by=['DATE', 'SIZE'])[i].transform('mean')

group_bus = group_bus.drop_duplicates(subset=['DATE', 'SIZE'], keep='first')

group_variables.remove('DATE')
group_variables.remove('SIZE')

for i in group_variables:
    del bus[i]

bus = pd.merge(bus, group_bus, how='left', on=['DATE', 'SIZE'])
bus = bus.drop_duplicates(subset=['DATE', 'SIZE'], keep='first')

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

bus.to_csv(STRING.PATH.file_processed, index=False, sep=';', encoding='latin1')

bus = pd.read_csv(STRING.PATH.file_processed, delimiter=';', encoding='latin1')
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
bus.to_csv(STRING.PATH.path_files + 'before_lags.csv', index=False, sep=';')
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
shifted_var = ['DAY_holiday', 'PASSENGER_SUM_DAY',
                              'PASSENGER_SUM_DAY_TRAIN',
                              'PASSENGER_SUM_DAY_METRO']
lags = 5
i = 1
id_shifter = list(bus['SIZE'].unique())
print(id_shifter)
bus_shifted = pd.DataFrame(data=None, columns=['DATE', 'SIZE']+shifted_var)
date_i = date.copy()
date_i = date_i.sort_values(by=['DATE'], ascending=[True])
date_i = date_i.drop_duplicates(subset=['DATE'])

for line in id_shifter:
    date_sub = date_i.copy()
    date_sub['SIZE'] = pd.Series(line, index=date_sub.index)
    print(line)
    bus_line_i = bus[bus['SIZE'] == line]
    bus_line_i = bus_line_i[['DATE', 'SIZE'] + shifted_var]
    bus_line_i = bus_line_i.drop_duplicates(subset=['DATE', 'SIZE'])
    bus_line_i = pd.merge(date_sub, bus_line_i, how='left', on=['DATE', 'SIZE'])
    while i <= lags:
        bus_shifted_lag = bus_line_i[shifted_var].shift(i)
        bus_shifted_lag = bus_shifted_lag.add_suffix('_L' + str(i))
        bus_line_i = pd.concat([bus_line_i, bus_shifted_lag], axis=1)
        i += 1
    i = 1
    bus_line_i = bus_line_i.reset_index(drop=True)
    bus_line_i.to_csv(STRING.PATH.path_files + 'line_' + str(line) + '.csv', index=False, sep=';')
    bus_shifted = pd.concat([bus_shifted, bus_line_i], axis=0, ignore_index=True)
    print(bus_shifted['DATE'].head(1))
    print(bus_shifted['DATE'].tail(1))
    print('TABLE SHAPE', bus_shifted.shape)
print(bus_shifted)
bus_shifted.to_csv(STRING.PATH.path_files + 'bus_shifted.csv', index=False, sep=';')
del bus_shifted
bus_shifted = pd.read_csv(STRING.PATH.path_files + 'bus_shifted.csv', sep=';', encoding='latin1')
bus_shifted = bus_shifted.dropna(subset=['SIZE'])
bus_shifted['DATE'] = pd.to_datetime(bus_shifted['DATE'])
bus_shifted['SIZE'] = bus_shifted['SIZE'].map(int)

bus = bus.drop(shifted_var, axis=1)

bus = pd.merge(bus, bus_shifted, how='left', on=['DATE', 'SIZE'])
bus = bus.dropna(axis=0, how='any')
bus.to_csv(STRING.PATH.path_files + 'bus_test.csv', sep=';', encoding='latin1', index=False)
del bus_shifted

# INTERACTIONS

key_df = bus[keys]
'''
for i in keys:
    del bus[i]

poly = pp.PolynomialFeatures(2, interaction_only=False)
output_array = poly.fit_transform(bus)
columns = poly.get_feature_names(bus.columns)
bus = pd.DataFrame(bus, columns=columns)
bus.to_csv(STRING.PATH.path_files + 'bus_poly.csv', index=False, sep=';')
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

# NORMALIZATION
# Normal
key_df = bus[keys]
for i in keys:
    del bus[i]


columns = bus.columns.values.tolist()

list_params = []
for i in columns:
    mean = bus[i].mean()
    std = bus[i].std()
    list_params.append([i, mean, std])

params = pd.DataFrame(list_params, columns=['VARIABLE', 'MEAN', 'STD'])
params.to_csv('params.csv', index=False, sep=';')

bus = scale(bus)
bus = pd.DataFrame(bus, columns=columns)
bus = pd.concat([bus, key_df], axis=1)

# Robust
# bus = procc_utils.robust_scale(bus, key_vars=keys)

# PCA REDUCTION
# bus = procc_utils.pca_reduction(bus, normalization=False)
date = date.sort_values(by=['DATE'], ascending=[True])
date = date.reset_index(drop=True)

date['TREND'] = pd.Series(date.index, index=date.index)
bus = pd.merge(bus, date, how='left', on='DATE')

bus.to_csv(STRING.PATH.path_files + STRING.PATH.file_final, sep=';', encoding='latin1', index=False)