import models
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import STRING
import utils.process_utils as putils
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statistics.basic_statistics import basic_statistics

df = pd.read_csv(STRING.PATH.file_final, delimiter=';', encoding='latin1')
variable_columns = df.columns.values.tolist()
variable_columns = [i for i in variable_columns if not i.startswith('INCOME')]

variable_columns = [i for i in variable_columns if not i.startswith('INTERN')]
variable_columns = [i for i in variable_columns if not i.startswith('DAY')]

variable_columns += ['DAY_holiday']

del_var = ['PASSENGER_MAX_DAY', 'PASSENGER_AVG_DAY', 'PASSENGER_MEDIAN_DAY', 'PASSENGER_MIN_DAY', 'PASSENGERS_METRO_TRAIN',
           'PASSENGERS_BUS_TRAIN', 'PASSENGERS_BUS_METRO', 'PASSENGER_SUM_DAY_METRO', 'PASSENGER_SUM_DAY_TRAIN',
           'YEAR', 'MONTH', 'DLS_MIN', 'EMAE_DESESTAC_VAR', 'DLS_MAX',
'EMAE','INDUSTRY','YEAR_2016','ACTIVITY_RATE_GBA',
            'DLS_OPEN','RATE_Jul14', 'YEAR_2015',
             'TURISM','UNEMPLOYMENT_RATE_GBA','YEAR_2014','PIB_REAL',
           'PIB_NOMINAL', 'EMAE_VAR','TRANSPORT', 'PUBLIC_SERVICES', 'EMAE_DESESTAC', 'CONSTRUCTION', 'WAGES_PRIVATE_FORMAL',
           'WAGES_PUBLIC','PRICE_VAR','WAGES_PRIVATE_INFORMAL', 'WAGES_GENERAL_VAR','COLD_UNITIES (h)',  'WEEKDAY_0',
           'WEEKDAY_1', 'WEEKDAY_2', 'WEEKDAY_3', 'WEEKDAY_4', 'MONTH_3', 'MONTH_4', 'MONTH_5',
           'MONTH_6',  'MONTH_8', 'MONTH_9', 'MONTH_10', 'MONTH_11', 'MONTH_12', 'WIND_SPEED_2m(km/h)'
           ]


'''
x = ['PASSENGER_SUM_DAY_L2', 'PASSENGER_SUM_DAY_TRAIN_L5', 'FARE_METRO_TRAIN',  'PASSENGER_SUM_DAY_TRAIN_L2', 
'PASSENGER_SUM_DAY_METRO_L3',  'PASSENGER_SUM_DAY_METRO_L2',  'DLS_LAST',  
'RATE_Apr16', 'FARE_AVG_METRO', 'PASSENGER_SUM_DAY_L1', 'WEEKDAY_5', 'PASSENGER_SUM_DAY_L4', 
 'TEMP_1.5m', 'FARE_BUS_METRO', 'PASSENGER_SUM_DAY_L3', 
 'HUMIDITY(%)', 'MONTH_holiday',  'PASSENGER_SUM_DAY_TRAIN_L4', 
'PASSENGER_SUM_DAY_L5', 'PASSENGER_SUM_DAY_TRAIN_L1',  'PASSENGER_SUM_DAY_METRO_L5', 
'PASSENGER_SUM_DAY_TRAIN_L3', 'WORKDAY', 'WEEKDAY_6',  'PASSENGER_SUM_DAY_METRO_L1', 
 'FARE_BUS_TRAIN', 'PASSENGER_SUM_DAY_METRO_L4', 'WAGE', 'DUMMY_GOVERN', 'FARE_AVG_TRAIN', 
'COLD_HOURS (h)', 'PRECIPITACION-MM', 'DATE', 'SIZE', 'TREND', 'DAY_holiday', 'WIND_STRONG', 'MONTH_7',, 'MONTH_1', 'MONTH_2',
'TEMP_GROUND_10cm', 'TEMP_MIN_OUTSIDE_0.5m', 'WIND_SPEED_10m(km/h)', 'TEMP_MAX_1.5m', 'TEMP_MIN_1.5m',
           'ATM_PRESSURE (hPa)', 'TEMP_MIN_OUTSIDE_1.5m', 'HUMIDITY_8_14_20(%)', 'DLS_VAR', 'DEW (ºC)','WIND_MAX_SPEED(km/h)',
           'PRECIPITATION_MAX_30min (mm)', 'WET_FOLIAGE(h)', 'COMMERCE','PRECIPITATION_7_7(mm)','STEAM_TENSION (hPa)',]

'''


for i in del_var:
    variable_columns.remove(i)


print(variable_columns)
df = df[variable_columns]

y = df[['PASSENGER_SUM_DAY']]
keys = df[['DATE']]
x = df.copy()
del x['PASSENGER_SUM_DAY']


x['REAL_WAGE'] = x['WAGE'] / x['PRICE_LEVEL']
x['REAL_FARE'] = x['FARE_AVG'] / x['PRICE_LEVEL']
x['GASTO_TRANSPORTE'] = x['REAL_FARE'] / x['REAL_WAGE']
#del x['WAGE']
#del x['PRICE_LEVEL']
print(x.columns.values.tolist())


# OLS MODEL
x_ols = x.copy()

x_ols = x_ols[['PASSENGER_SUM_DAY_L1',  'WORKDAY', 'DAY_holiday',
               'COLD_HOURS (h)', 'PRECIPITACION-MM',  'HUMIDITY(%)',
                'TEMP_1.5m',
            'PRECIPITATION_MAX_30min (mm)',
                'REAL_WAGE', 'FARE_AVG',
                'FARE_BUS_TRAIN',
                'SIZE', 'TREND', 'WIND_STRONG', 'MONTH_1', 'MONTH_7'
  ]]



# x_ols['CONSTANT'] = pd.Series(1, index=x_ols.index)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def OLS(Y, x, vif=True):
    if vif:
        vif = pd.DataFrame()
        vif['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        vif['features'] = x.columns
        print(vif)


    mod = sm.OLS(Y, x.astype(float))
    res = mod.fit()
    result = res.summary()
    print(result)

OLS(y, x_ols)
x_ols = x_ols.reset_index(drop=True)
y = y.reset_index(drop=True)
# basic_statistics.errores_summ(x_ols.values, y.values)

# from models.stepwise_regression import stepwise_regression
# stepwise_regression.setpwise_reg(x.drop(['DATE'], axis=1).values, y.values, x.drop(['DATE'],axis=1).columns.values.tolist())


###############NET REGRESSION###########################################################################################
from models.glmnet_regression import glmnet_regression
#glmnet_regression.enet_reg_alpha(x_ols.values,y.values)
alphaStar = 0.00985865375619
lambdaStar = 1
#glmnet_regression.elastic_reg_betas(x_ols.values,y.values,x_ols.columns.values.tolist(),alphaStar)
#glmnet_regression.enet_r2(x_ols.values,y.values,alphaStar,lambdaStar)


############RANDOM FOREST###########################################################################################
#from models.random_forest import random_forest
#random_forest.random_forest(x_ols,y,x_ols.columns.values.tolist())

from models.gradient_boosting import gradient_boosting
gradient_boosting.gradient_boosting(x_ols.values,y.values,x_ols.columns.values.tolist(), output_pred='pred.csv')
##########BAGGING#######################################################################################################
# from models.bagging import bagging
# print(x_ols.columns.values.tolist())
# bagging.bagging(x_ols.values,y.values,None, x_ols.columns.values.tolist())

##########BINARY DECISION TREE#########################################################################################

from models.binary_decisiontree import binary_decisiontree
import matplotlib.pyplot as plot
#binary_decisiontree.binary_decision_tree_depth(x_ols.values,y.values)
'''
mse_list = []
for i in range(1, 31, 1):
    mse = binary_decisiontree.binary_decision_tree(x_ols,y,i, x_ols.columns.values.tolist(), plot=False)
    mse_list.append(mse)
plot.plot(range(1,31,1), mse_list)
plot.axis('tight')
plot.xlabel('Tree Depth')
plot.ylabel('MSE')
plot.show()
'''
#binary_decisiontree.binary_decision_tree(x_ols,y,30, x_ols.columns.values.tolist(), plot_fig=True)


# ERT PERFORMANCE
Train, Valid, Test = putils.training_test_valid(x_ols, y)
'''
max_depth, n_estimators, bootstrap, oob_score, max_features, min_sample_leaf, min_sample_split, \
max_features = models.extreme_randomize.tunning(Train, Valid, label='PASSENGER_SUM_DAY', key=None)
'''
max_depth = None
n_estimators = 1000
bootstrap = True
oob_score = False
max_features = 6
min_sample_leaf = round((len(Train.index)) * 0.01)
min_sample_split = min_sample_leaf * 10
'''
models.extreme_randomize.evaluation(Train, Test, Valid, max_depth, n_estimators, bootstrap, oob_score, max_features,
                                    min_sample_leaf, min_sample_split, label='PASSENGER_SUM_DAY', key=None)


# ELASTICITY BY DATE

Train, Test_Date = putils.training_test_valid_by_date(x, y, test_date='RATE_Apr16')

max_depth, n_estimators, bootstrap, oob_score, max_features, min_sample_leaf, min_sample_split, \
max_features = models.extreme_randomize.tunning(Train, Valid)


Valid = None
models.extreme_randomize.evaluation(Train, Test_Date, Valid, max_depth, n_estimators, bootstrap, oob_score, max_features,
                                    min_sample_leaf, min_sample_split, label='PASSENGER_SUM_DAY', output_prob=True)
'''