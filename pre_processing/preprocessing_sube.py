import STRING
from utils import merge_files, read_csv
import os
from zipfile import ZipFile
import pandas as pd


os.chdir(STRING.PATH.path_files)


def extracting_zip():
    '''
    with ZipFile(STRING.PATH.zip_file, 'r') as zf:
        zf.extractall(path=STRING.PATH.path_files, pwd=STRING.PATH.zip_pass.encode('cp850','replace'))
    '''
    merge_files.merge_files(os.getcwd(), os.getcwd(), 'UsosSubtes', STRING.PATH.file_subte, folder_name='')
    merge_files.merge_files(os.getcwd(), os.getcwd(), 'UsosTrenes', STRING.PATH.file_tren, folder_name='')
    #merge_files.merge_files(os.getcwd(), os.getcwd(), 'Entrega_dggi_hora', STRING.PATH.file_bus, folder_name='')

    '''
    file_list = [f for f in os.listdir(os.getcwd())]
    file_list = [f for f in file_list if not '.zip' in f or not 'merged' in f]
    for i in file_list:
        os.remove(i)
    '''

def preprocessing_subte(subte: pd.DataFrame, del_hour=True, by = ['ID_LINE', 'DATE']):
    # Rename the variables
    dict_rename_subte = {'ID_EMPRESA': 'ID_COMPANY', 'ID_LINEA': 'ID_LINE', 'DIA_TRX': 'DATE', 'HORA_TRX': 'HOUR',
                         'NOMBRE': 'NAME', 'CANT_USOS': 'PASSENGERS', 'MON_USOS': 'INCOME'}

    df = subte.rename(columns=dict_rename_subte)
    df = df[df['INCOME'] >= 0]

    # Delete unnecessary variables
    del_var = ['ID_COMPANY', 'NAME']
    for i in del_var:
        del df[i]

    # Convert date to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

    if 'ID_LINE' not in by:
        del df['ID_LINE']
        df = df.groupby(by=['DATE', 'HOUR'])['PASSENGERS', 'INCOME'].sum().reset_index()

    if del_hour == True:
        del df['HOUR']

    # We calculate statistics about the number of passengers-income in one hour per day
    max_pass = df.copy()

    max_pass['PASSENGER_MAX_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform(max)
    max_pass['PASSENGER_MIN_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform(min)
    max_pass['PASSENGER_MEDIAN_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('median')
    max_pass['PASSENGER_AVG_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('mean')
    max_pass['PASSENGER_SUM_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('sum')
    max_pass['INCOME_MAX_DAY'] = max_pass.groupby(by=by)['INCOME'].transform(max)
    max_pass['INCOME_MIN_DAY'] = max_pass.groupby(by=by)['INCOME'].transform(min)
    max_pass['INCOME_MEDIAN_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('median')
    max_pass['INCOME_AVG_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('mean')
    max_pass['INCOME_SUM_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('sum')
    del max_pass['PASSENGERS'], max_pass['INCOME']
    max_pass = max_pass.drop_duplicates()

    # GROUP BY ID-LINE, DATE
    df = df.groupby(by=by)['PASSENGERS', 'INCOME'].sum().reset_index()

    # We merge with max_pass
    df = pd.merge(df, max_pass, how='left', on=by)

    # We calculate the average fare
    df['FARE_AVG'] = df['INCOME'] / df['PASSENGERS']

    # DATEs Variables
    date_var = ['DATE']
    year = 'YEAR'
    month = 'MONTH'
    day = 'DAY'
    weekday = 'WEEKDAY'
    for i in date_var:
        df[year] = pd.DatetimeIndex(df[i]).year
        df[month] = pd.DatetimeIndex(df[i]).month
        df[day] = pd.DatetimeIndex(df[i]).day
        df[weekday] = pd.Series(df[i].dt.weekday, index=df.index)
        
    # Dummy Year
    dummy_year = pd.get_dummies(df[year], prefix=year)
    df = pd.concat([df, dummy_year], axis=1)
    del dummy_year
    
    # Dummy Month
    dummy_month = pd.get_dummies(df[month], prefix=month)
    df = pd.concat([df, dummy_month], axis=1)
    del dummy_month
    
    # Dummy day
    dummy_day = pd.get_dummies(df[day], prefix=day)
    df = pd.concat([df, dummy_day], axis=1)
    del dummy_day
    
    # Dummy weekday
    dummy_weekday = pd.get_dummies(df[weekday], prefix=weekday)
    df = pd.concat([df, dummy_weekday], axis=1)
    del dummy_weekday
    
    # By month day
    day_1_10 = day + '1_10'
    day_10_20 = day + '_10_20'
    day_20_30 = day + '20_30'
    df[day_1_10] = pd.Series(0, index=df.index)
    df[day_10_20] = pd.Series(0, index=df.index)
    df[day_20_30] = pd.Series(0, index=df.index)
    df.loc[df[day].between(1, 10, True), day_1_10] = 1
    df.loc[df[day].between(11, 20, True), day_10_20] = 1
    df.loc[df[day].between(21, 31, True), day_20_30] = 1
    
    # Holidays
    month_holiday = month + '_holiday'
    df[month_holiday] = pd.Series(0, index=df.index)
    df.loc[df[month].isin([1, 2]), month_holiday] = 1

    # National Days
    day_holiday = day + '_holiday'
    df[day_holiday] = pd.Series(0, index=df.index)
    for i in STRING.dates.feriados:
        df.loc[df['DATE'] == i, day_holiday] = 1

    # UP RATE
    df['RATE_Jan13'] = pd.Series(0, index=df.index)
    df['RATE_Jan14'] = pd.Series(0, index=df.index)
    df['RATE_Jul14'] = pd.Series(0, index=df.index)
    df['RATE_Apr16'] = pd.Series(0, index=df.index)
    df.loc[df['DATE'] < STRING.fares.date_fare_jan_14, 'RATE_Jan13'] = 1
    df.loc[(df['DATE'] >= STRING.fares.date_fare_jan_14)&(df['DATE'] < STRING.fares.date_fare_jul_14), 'RATE_Jan14'] = 1
    df.loc[(df['DATE'] >= STRING.fares.date_fare_jul_14)&(df['DATE'] < STRING.fares.date_fare_apr_16), 'RATE_Jul14'] = 1
    df.loc[df['DATE'] >= STRING.fares.date_fare_apr_16, 'RATE_Apr16'] = 1

    # WORKDAY
    df['WORKDAY'] = pd.Series(0, index=df.index)
    df.loc[df[weekday].isin([0, 1, 2, 3, 4]), 'WORKDAY'] = 1

    # DUMMY Vars: LINE, WEEKDAY, MONTH,
    for i in [year, month, day, weekday]:
        del df[i]

    df.to_csv(STRING.PATH.file_preprocessed_subte, sep=';', index=False, encoding='latin1')

    
 


def preprocessing_tren(tren: pd.DataFrame, del_hour=True, by=['ID_LINE', 'DATE']):

    # Rename the variables
    dict_rename_tren = {'ID_EMPRESA': 'ID_LINE', 'DIA_TRX': 'DATE', 'HORA_TRX': 'HOUR', 'NOMBRE': 'NAME',
                        'CANT_USOS': 'PASSENGERS', 'MON_USOS': 'INCOME'}

    df = tren.rename(columns=dict_rename_tren)
    df = df[df['INCOME'] >= 0]

    # Delete unnecessary variables
    del_var = ['NAME']
    for i in del_var:
        del df[i]

    # Convert date to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

    if 'ID_LINE' not in by:
        del df['ID_LINE']
        df = df.groupby(by=['DATE', 'HOUR'])['PASSENGERS', 'INCOME'].sum().reset_index()

    if del_hour == True:
        del df['HOUR']

    # We calculate statistics about the number of passengers-income in one hour per day
    max_pass = df.copy()

    max_pass['PASSENGER_MAX_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform(max)
    max_pass['PASSENGER_MIN_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform(min)
    max_pass['PASSENGER_MEDIAN_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('median')
    max_pass['PASSENGER_AVG_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('mean')
    max_pass['PASSENGER_SUM_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('sum')
    max_pass['INCOME_MAX_DAY'] = max_pass.groupby(by=by)['INCOME'].transform(max)
    max_pass['INCOME_MIN_DAY'] = max_pass.groupby(by=by)['INCOME'].transform(min)
    max_pass['INCOME_MEDIAN_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('median')
    max_pass['INCOME_AVG_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('mean')
    max_pass['INCOME_SUM_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('sum')
    del max_pass['PASSENGERS'], max_pass['INCOME']
    max_pass = max_pass.drop_duplicates()

    # GROUP BY ID-LINE, DATE
    df = df.groupby(by=by)['PASSENGERS', 'INCOME'].sum().reset_index()

    # We merge with max_pass
    df = pd.merge(df, max_pass, how='left', on=by)

    # We calculate the average fare
    df['FARE_AVG'] = df['INCOME'] / df['PASSENGERS']

    # DATEs Variables
    date_var = ['DATE']
    year = 'YEAR'
    month = 'MONTH'
    day = 'DAY'
    weekday = 'WEEKDAY'
    for i in date_var:
        df[year] = pd.DatetimeIndex(df[i]).year
        df[month] = pd.DatetimeIndex(df[i]).month
        df[day] = pd.DatetimeIndex(df[i]).day
        df[weekday] = pd.Series(df[i].dt.weekday, index=df.index)

    # Dummy Year
    dummy_year = pd.get_dummies(df[year], prefix=year)
    df = pd.concat([df, dummy_year], axis=1)
    del dummy_year

    # Dummy Month
    dummy_month = pd.get_dummies(df[month], prefix=month)
    df = pd.concat([df, dummy_month], axis=1)
    del dummy_month

    # Dummy day
    dummy_day = pd.get_dummies(df[day], prefix=day)
    df = pd.concat([df, dummy_day], axis=1)
    del dummy_day

    # Dummy weekday
    dummy_weekday = pd.get_dummies(df[weekday], prefix=weekday)
    df = pd.concat([df, dummy_weekday], axis=1)
    del dummy_weekday

    # By month day
    day_1_10 = day + '1_10'
    day_10_20 = day + '_10_20'
    day_20_30 = day + '20_30'
    df[day_1_10] = pd.Series(0, index=df.index)
    df[day_10_20] = pd.Series(0, index=df.index)
    df[day_20_30] = pd.Series(0, index=df.index)
    df.loc[df[day].between(1, 10, True), day_1_10] = 1
    df.loc[df[day].between(11, 20, True), day_10_20] = 1
    df.loc[df[day].between(21, 31, True), day_20_30] = 1

    # Holidays
    month_holiday = month + '_holiday'
    df[month_holiday] = pd.Series(0, index=df.index)
    df.loc[df[month].isin([1, 2, 7]), month_holiday] = 1

    # National Days
    day_holiday = day + '_holiday'
    df[day_holiday] = pd.Series(0, index=df.index)
    for i in STRING.dates.feriados:
        df.loc[df['DATE'] == i, day_holiday] = 1

    # UP RATE
    df['RATE_Jan13'] = pd.Series(0, index=df.index)
    df['RATE_Jan14'] = pd.Series(0, index=df.index)
    df['RATE_Jul14'] = pd.Series(0, index=df.index)
    df['RATE_Apr16'] = pd.Series(0, index=df.index)
    df.loc[df['DATE'] < STRING.fares.date_fare_jan_14, 'RATE_Jan13'] = 1
    df.loc[
        (df['DATE'] >= STRING.fares.date_fare_jan_14) & (df['DATE'] < STRING.fares.date_fare_jul_14), 'RATE_Jan14'] = 1
    df.loc[
        (df['DATE'] >= STRING.fares.date_fare_jul_14) & (df['DATE'] < STRING.fares.date_fare_apr_16), 'RATE_Jul14'] = 1
    df.loc[df['DATE'] >= STRING.fares.date_fare_apr_16, 'RATE_Apr16'] = 1

    # WORKDAY
    df['WORKDAY'] = pd.Series(0, index=df.index)
    df.loc[df[weekday].isin([0, 1, 2, 3, 4]), 'WORKDAY'] = 1

    # DUMMY Vars: LINE, WEEKDAY, MONTH,
    for i in [year, month, day, weekday]:
        del df[i]

    df.to_csv(STRING.PATH.file_preprocessed_tren, sep=';', index=False, encoding='latin1')



def preprocessing_bus(bus : pd.DataFrame, del_hour=True, by=['ID_LINE', 'DATE']):
    # Rename the variables
    dict_rename_bus = {'ID_EMPRESA': 'ID_COMPANY', 'ID_LINEA': 'ID_LINE', 'INTERNO': 'INTERN', 'RAMAL': 'BRANCH',
                       'DIA_TRX': 'DATE', 'HORA_TRX': 'HOUR', 'CANTIDAD_USOS': 'PASSENGERS', 'MONTO': 'INCOME'}

    df = bus.rename(columns=dict_rename_bus)
    df = df[df['INCOME'] >= 0]

    # Convert date to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

    # Delete unnecessary variables
    del_var = ['ID_COMPANY', 'BRANCH']
    for i in del_var:
        del df[i]

    # First we count the number of INTERNs by LINE by HOUR
    if 'ID_LINE' not in by:
        del df['ID_LINE']
        df = df.groupby(by=['DATE', 'HOUR', 'INTERN'])['PASSENGERS', 'INCOME'].sum().reset_index()

    df_intern = df.copy()
    df_intern = df_intern[['DATE', 'HOUR', 'ID_LINE', 'INTERN']]
    df_intern = df_intern.drop_duplicates()
    df_intern['INTERN_COUNT_HOUR'] = df_intern.groupby(by=by+['HOUR'])['INTERN'].transform('count')
    df_intern['INTERN_MAX_HOUR'] = df_intern.groupby(by=by)['INTERN_COUNT_HOUR'].transform(max)
    df_intern['INTERN_MIN_HOUR'] = df_intern.groupby(by=by)['INTERN_COUNT_HOUR'].transform(min)
    df_intern['INTERN_MEDIAN_HOUR'] = df_intern.groupby(by=by)['INTERN_COUNT_HOUR'].transform('median')
    df_intern['INTERN_AVG_HOUR'] = df_intern.groupby(by=by)['INTERN_COUNT_HOUR'].transform('mean')

    del df_intern['HOUR'], df_intern['INTERN'], df_intern['INTERN_COUNT_HOUR']
    df_intern = df_intern.drop_duplicates()

    if del_hour == True:
        del df['HOUR']

    # We calculate statistics about the number of passengers-income in a day
    max_pass = df.copy()

    max_pass['PASSENGER_MAX_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform(max)
    max_pass['PASSENGER_MIN_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform(min)
    max_pass['PASSENGER_MEDIAN_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('median')
    max_pass['PASSENGER_AVG_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('mean')
    max_pass['PASSENGER_SUM_DAY'] = max_pass.groupby(by=by)['PASSENGERS'].transform('sum')
    max_pass['INCOME_MAX_DAY'] = max_pass.groupby(by=by)['INCOME'].transform(max)
    max_pass['INCOME_MIN_DAY'] = max_pass.groupby(by=by)['INCOME'].transform(min)
    max_pass['INCOME_MEDIAN_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('median')
    max_pass['INCOME_AVG_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('mean')
    max_pass['INCOME_SUM_DAY'] = max_pass.groupby(by=by)['INCOME'].transform('sum')
    del max_pass['PASSENGERS'], max_pass['INCOME']
    max_pass = max_pass.drop_duplicates()

    # GROUP BY ID-LINE, DATE
    if 'ID_LINE' not in by:
        del df['ID_LINE']
        del max_pass['ID_LINE']
        del df_intern['ID_LINE']
        
    # WE CALCULATE THE SIZE BY DAY OF THE LINES
    intern_day = df.copy()
    intern_day = intern_day[['DATE', 'ID_LINE', 'INTERN']]
    intern_day = intern_day.drop_duplicates()
    intern_day['INTERN_COUNT_DAY'] = intern_day.groupby(by=by)['INTERN'].transform('count')
    del intern_day['INTERN']
    intern_day = intern_day.drop_duplicates()
    
    
    
    # WE GROUP AS DAY AND LINE
    df = df.groupby(by=by)['PASSENGERS', 'INCOME'].sum().reset_index()

    # We merge with max_pass
    df = pd.merge(df, max_pass, how='left', on=by)
    df = pd.merge(df, df_intern, how='left', on=by)
    df = pd.merge(df, intern_day, how='left', on=by)

    # We calculate the average fare
    df['FARE_AVG'] = df['INCOME'] / df['PASSENGERS']

    # DATEs Variables
    date_var = ['DATE']
    year = 'YEAR'
    month = 'MONTH'
    day = 'DAY'
    weekday = 'WEEKDAY'
    for i in date_var:
        df[year] = pd.DatetimeIndex(df[i]).year
        df[month] = pd.DatetimeIndex(df[i]).month
        df[day] = pd.DatetimeIndex(df[i]).day
        df[weekday] = pd.Series(df[i].dt.weekday, index=df.index)

    # Dummy Year
    dummy_year = pd.get_dummies(df[year], prefix=year)
    df = pd.concat([df, dummy_year], axis=1)
    del dummy_year

    # Dummy Month
    dummy_month = pd.get_dummies(df[month], prefix=month)
    df = pd.concat([df, dummy_month], axis=1)
    del dummy_month

    # Dummy day
    dummy_day = pd.get_dummies(df[day], prefix=day)
    df = pd.concat([df, dummy_day], axis=1)
    del dummy_day

    # Dummy weekday
    dummy_weekday = pd.get_dummies(df[weekday], prefix=weekday)
    df = pd.concat([df, dummy_weekday], axis=1)
    del dummy_weekday

    # By month day
    day_1_10 = day + '1_10'
    day_10_20 = day + '_10_20'
    day_20_30 = day + '20_30'
    df[day_1_10] = pd.Series(0, index=df.index)
    df[day_10_20] = pd.Series(0, index=df.index)
    df[day_20_30] = pd.Series(0, index=df.index)
    df.loc[df[day].between(1, 10, True), day_1_10] = 1
    df.loc[df[day].between(11, 20, True), day_10_20] = 1
    df.loc[df[day].between(21, 31, True), day_20_30] = 1

    # Holidays
    month_holiday = month + '_holiday'
    df[month_holiday] = pd.Series(0, index=df.index)
    df.loc[df[month].isin([1, 2, 7]), month_holiday] = 1

    # National Days
    day_holiday = day + '_holiday'
    df[day_holiday] = pd.Series(0, index=df.index)
    for i in STRING.dates.feriados:
        df.loc[df['DATE'] == i, day_holiday] = 1

    # UP RATE
    df['RATE_Jan13'] = pd.Series(0, index=df.index)
    df['RATE_Jan14'] = pd.Series(0, index=df.index)
    df['RATE_Jul14'] = pd.Series(0, index=df.index)
    df['RATE_Apr16'] = pd.Series(0, index=df.index)
    df.loc[df['DATE'] < STRING.fares.date_fare_jan_14, 'RATE_Jan13'] = 1
    df.loc[
        (df['DATE'] >= STRING.fares.date_fare_jan_14) & (df['DATE'] < STRING.fares.date_fare_jul_14), 'RATE_Jan14'] = 1
    df.loc[
        (df['DATE'] >= STRING.fares.date_fare_jul_14) & (df['DATE'] < STRING.fares.date_fare_apr_16), 'RATE_Jul14'] = 1
    df.loc[df['DATE'] >= STRING.fares.date_fare_apr_16, 'RATE_Apr16'] = 1

    # WORKDAY
    df['WORKDAY'] = pd.Series(0, index=df.index)
    df.loc[df[weekday].isin([0, 1, 2, 3, 4]), 'WORKDAY'] = 1

    # DUMMY Vars: LINE, WEEKDAY, MONTH,
    for i in [year, month, day, weekday]:
        del df[i]
    del df['INTERN']
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df

if __name__ == '__main__':
    # extracting_zip()
    '''
    tren = read_csv.reading_files_transporte(STRING.PATH.file_tren)
    preprocessing_tren(tren, by=['DATE'])

    subte = read_csv.reading_files_transporte(STRING.PATH.file_subte)
    preprocessing_subte(subte, by=['DATE'])
    '''

    file_list = [f for f in os.listdir(os.getcwd())]
    file_list = list(set([f for f in file_list if 'Entrega_dggi_' in f]))
    df = pd.read_csv(file_list[0], sep=';', encoding='latin1')
    df = preprocessing_bus(df)

    del file_list[0]
    for i in file_list:
        print(i)
        df_i = pd.read_csv(i, sep=';', encoding='latin1')
        df_i = preprocessing_bus(df_i)
        df = pd.concat([df, df_i], axis=0)
        del df_i


    df.to_csv(STRING.PATH.file_preprocessed_bus, sep=';', index=False, encoding='latin1')




