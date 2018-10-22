import pandas as pd
import STRING
from utils.statistics_temporal import test_stationarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot
import seaborn as sns

df = pd.read_csv(STRING.PATH.final_file, sep=';', encoding='latin1', parse_dates=['DATE'])
np.random.seed(42)
df = df.sort_values(by=['DATE'], ascending=[True])
df = df[(df['RATE_Jul14'] == 1)|(df['RATE_Apr16'] == 1)]
del df['RATE_Jan14']
df = df.set_index('DATE')

# Stationarity
test_stationarity(df['PASSENGER_SUM_DAY'], plot_show=True, plot_name='passengers_day.png')
# if the ‘Test Statistic’ is greater than the ‘Critical Value’ than the time series is stationary.
# df['PASSENGER_SUM_DAY'] = df['PASSENGER_SUM_DAY'] - df['PASSENGER_SUM_DAY'].shift(1)

need_differentiate = df.columns.values.tolist()
need_differentiate = [x for x in need_differentiate if not (x.startswith('DAY') or x.startswith('WEEKDAY') or x.startswith('MONTH') or x.startswith('FARE') or x in 'PASSENGER_SUM_DAY' or x in 'TREND')]
for i in need_differentiate:
    t_value, critical_value = test_stationarity(df[i], plot_show=False)
    if t_value < critical_value:
        name = 'D_' + str(i)
        print(name)
        df[name] = df[i] - df[i].shift(1)
        del df[i]

# LAGS
lags = 2
for col in df.columns.values.tolist():
    if col != 'RATE_Apr16':
        for i in range(1, lags + 1, 1):
            df['L_' + str(i) + '_' + col] = df[col].shift(i)

# WE ORDER THE LAGS
cols = df.columns.values.tolist()
df = df[['PASSENGER_SUM_DAY'] + [x for x in cols if x.startswith('L_')] + ['FARE_AVG', 'RATE_Apr16']]
df = df.dropna()
df = df.reset_index(drop=False)

# NORMALIZE
df_c = df[['DATE', 'RATE_Apr16', 'FARE_AVG']]
scaler = MinMaxScaler(feature_range=(-1, 1))
cols = df.drop(['DATE', 'FARE_AVG', 'RATE_Apr16'], axis=1).values.tolist()
df = scaler.fit_transform(df.drop(['DATE', 'FARE_AVG', 'RATE_Apr16'], axis=1))
df = pd.DataFrame(df, columns=cols)
df = pd.concat([df_c, df], axis=1)

# SPLIT DATA
normal = df[df['RATE_Apr16'] == 0]
anormal = df[df['RATE_Apr16'] == 1]
normal = normal.drop(['DATE', 'FARE_AVG', 'RATE_Apr16'], axis=1)
anormal = anormal.drop(['DATE', 'FARE_AVG', 'RATE_Apr16'], axis=1)
split = int(len(normal.index)*0.70)

train_normal = normal.loc[0:split + 1]
print(train_normal.shape)
valid_normal = normal.loc[split + 1:]
test_normal = valid_normal.loc[valid_normal.index[0] + int(len(valid_normal.index)*0.5) + 1:]
valid_normal = valid_normal.loc[:valid_normal.index[0] + int(len(valid_normal.index)*0.5) + 1]
test = anormal.copy()

# DATA - [SAMPLES - TIME STEP - FEATURES]
train_normal_x, train_normal_y = train_normal.values[:, 1:], train_normal.values[:, 1]
valid_normal_x, valid_normal_y = valid_normal.values[:, 1:], valid_normal.values[:, 1]
test_normal_x, test_normal_y = test_normal.values[:, 1:], test_normal.values[:, 1]
test_x, test_y = test.values[:, 1:], test.values[:, 1]

train_normal_x = np.reshape(train_normal_x, (train_normal_x.shape[0], 1, train_normal_x.shape[1]))
valid_normal_x = np.reshape(valid_normal_x, (valid_normal_x.shape[0], 1, valid_normal_x.shape[1]))
test_normal_x = np.reshape(test_normal_x, (test_normal_x.shape[0], 1, test_normal_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
print(train_normal_x.shape, train_normal_y.shape, valid_normal_x.shape, valid_normal_y.shape)
# MODEL
model = Sequential()
model.add(LSTM(100, input_shape=(1, train_normal_x.shape[2]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adagrad')
print("Inputs: " + str(model.input_shape))
print("Outputs: " + str(model.output_shape))
print("Actual input: " + str(train_normal_x.shape))
print("Actual output:" + str(train_normal_y.shape))
early_stopping = EarlyStopping(patience=2)
model.fit(train_normal_x, train_normal_y, epochs=100, batch_size=50, verbose=2, callbacks=[early_stopping],
          shuffle=False,
          validation_data=(valid_normal_x, valid_normal_y))

# PREDICT
# Train
train_predict = model.predict(train_normal_x)
train_normal_x = train_normal_x.reshape(train_normal_x.shape[0], train_normal_x.shape[2])
train_predict = np.concatenate((train_predict, train_normal_x), axis=1)
inv_predict = scaler.inverse_transform(train_predict)
train_predict = inv_predict[:, 0]

train_normal_y = train_normal_y.reshape(len(train_normal_y), 1)
inv_y = np.concatenate((train_normal_y, train_normal_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
train_true = inv_y[:, 0]

# Valid
valid_predict = model.predict(valid_normal_x)
valid_normal_x = valid_normal_x.reshape(valid_normal_x.shape[0], valid_normal_x.shape[2])
valid_predict = np.concatenate((valid_predict, valid_normal_x), axis=1)
inv_predict = scaler.inverse_transform(valid_predict)
valid_predict = inv_predict[:, 0]

valid_normal_y = valid_normal_y.reshape(len(valid_normal_y), 1)
inv_y = np.concatenate((valid_normal_y, valid_normal_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
valid_true = inv_y[:, 0]

# Test Normal
test_normal_predict = model.predict(test_normal_x)
test_normal_x = test_normal_x.reshape(test_normal_x.shape[0], test_normal_x.shape[2])
test_normal_predict = np.concatenate((test_normal_predict, test_normal_x), axis=1)
inv_predict = scaler.inverse_transform(test_normal_predict)
test_normal_predict = inv_predict[:, 0]

test_normal_y = test_normal_y.reshape(len(test_normal_y), 1)
inv_y = np.concatenate((test_normal_y, test_normal_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
test_normal_true = inv_y[:, 0]

# Test
test_predict = model.predict(test_x)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])
test_predict = np.concatenate((test_predict, test_x), axis=1)
inv_predict = scaler.inverse_transform(test_predict)
test_predict = inv_predict[:, 0]

test_y = test_y.reshape(len(test_y), 1)
inv_y = np.concatenate((test_y, test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
test_true = inv_y[:, 0]

avg_rate_0 = df_c.loc[df_c['RATE_Apr16'] == 0, 'FARE_AVG'].mean()
avg_rate_1 = df_c.loc[df_c['RATE_Apr16'] == 1, 'FARE_AVG'].mean()

# PLOT
true_values = np.concatenate((train_true, valid_true, test_normal_true, test_true), axis=0)
true_values = pd.DataFrame(true_values, columns=['true_values'])

train_predict = pd.DataFrame(train_predict, columns=['predict'])
valid_predict = pd.DataFrame(valid_predict, columns=['predict'])
test_normal_predict = pd.DataFrame(test_normal_predict, columns=['predict'])
test_predict = pd.DataFrame(test_predict, columns=['predict'])

train_true = pd.DataFrame(train_true, columns=['true_values'])
valid_true = pd.DataFrame(valid_true, columns=['true_values'])
test_normal_true = pd.DataFrame(test_normal_true, columns=['true_values'])
test_true = pd.DataFrame(test_true, columns=['true_values'])

valid_true.index += len(train_true.index)
test_normal_true.index += len(train_true.index) + len(valid_true.index)
test_true.index += len(train_true.index) + len(valid_true.index) + len(test_normal_true.index)

train_predict.index += 1
valid_predict.index += len(train_predict.index) + 1
test_normal_predict.index += len(train_predict.index) + len(valid_predict.index) + 1
test_predict.index += len(train_predict.index) + len(valid_predict.index) + len(test_normal_predict.index) + 1

predict = pd.concat([train_predict, valid_predict, test_normal_predict, test_predict], axis=0)
rmse = np.sqrt((predict['predict'] - true_values['true_values'])**2)
rmse = rmse.dropna()
elasticity = (test_true['true_values'] - test_predict['predict']) / (avg_rate_0 - avg_rate_1)

f = plot.figure(figsize=(20, 10))
plot.subplots_adjust(hspace=0.01)
df_c['DATE'] = df_c['DATE'].astype(str)
threshold = df_c.index[df_c['DATE'] =='2016-04-08'].tolist()
values_date = np.arange(0, len(df_c.index), 50)
df_dates_xticks = df_c[df_c.index.isin(values_date)]
ax1 = plot.subplot(311)
ax1.plot(true_values, label='true_values')
ax1.plot(train_predict, label='train_predict')
ax1.plot(valid_predict, label='valid_predict')
ax1.plot(test_normal_true, label='test_normal_predict')
ax1.plot(test_predict, label='predicted_values')
ax1.axvline(threshold[0], ls='dashed', label='new Fare', color=sns.xkcd_rgb["dark teal"])
ax1.legend(bbox_to_anchor=(1.02, .3), borderaxespad=0., frameon=True)

ax2 = plot.subplot(312)
ax2.plot(rmse, label='RMSE')
ax2.axvline(threshold[0], ls='dashed', label='new Fare', color=sns.xkcd_rgb["dark teal"])
ax2.legend(bbox_to_anchor=(1, .3), borderaxespad=0., frameon=True)

ax3 = plot.subplot(313)
ax3.plot((true_values['true_values']-predict['predict']), label='Elasticity')
ax3.axvline(threshold[0],  ls='dashed', label='new Fare', color=sns.xkcd_rgb["dark teal"])
ax3.set_xticklabels(df_dates_xticks['DATE'], rotation=30, fontsize=8)
plot.xticks(values_date)
plot.title('Demand Prediction')
plot.show()

# PASSENGER LOSS
# MSE
mse_train = mean_squared_error(train_true['true_values'], train_predict['predict'])
mse_test = mean_squared_error(test_true['true_values'], test_predict['predict'])
print('RMSE TRAIN %.2f' % np.sqrt(mse_train))
print('RMSE TEST %.2f' % np.sqrt(mse_test))
print('PASSENGER LOSS NORMAL', (test_normal_true['true_values'].sum() - test_normal_predict['predict'].sum()) / len(test_normal.index))
print('PASSENGER LOSS', (test_true['true_values'].sum() - test_predict['predict'].sum()) / len(test.index))
print('PASSENGER ELASTICITY', np.mean(((test_predict['predict'] - test_true['true_values']) / test_true['true_values'])/ ((avg_rate_1 - avg_rate_0)/avg_rate_0)))

