import pandas as pd

def fillna_multioutput(df: pd.DataFrame, not_consider: list = ['DATE'], n_estimator=300,
                       max_depth=500, n_features=3, on='DATE'):
    """
    Multioutput regression used for estimating NaN values columns. 
    :return: df with multioutput fillna
    """

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split

    # First we determine which columns have NaN values are which not

    jcols = df.columns[df.isnull().any()].tolist()
    icols = df.columns.values.tolist()
    for i in jcols:
        icols.remove(i)

    # We evaluate here which rows are null value. This returns a boolean for each row
    notnans = df[jcols].notnull().all(axis=1)

    # We create a df with not nan values which will be used as train-test in a supervised model
    df_notnans = df[notnans]
    pd.set_option('display.max_rows', 350000)

    print('COLUMNAS SIN NAN VALUES')
    print(icols)
    print('COLUMNAS CON NAN VALUES')
    print(jcols)

    # We create a train-test set with X = icols values that do not have null values. And we try to estimate
    # the values of jcols (the columns with NaN). Here we are not considering the NaN values so we can estimate
    # as a supervised model the nan_cols. And finally, we apply the model estimation to the real NaN values.
    X_train, X_test, y_train, y_test = train_test_split(df_notnans[icols], df_notnans[jcols],
                                                        train_size=0.70,
                                                        random_state=42)

    n_estimator = n_estimator
    max_features = (round((len(df_notnans.columns)) / n_features))
    min_samples_leaf = round(len(df_notnans.index) * 0.005)
    min_samples_split = min_samples_leaf * 10
    max_depth = max_depth

    print('RANDOM FOREST WITH: ne_estimator=' + str(n_estimator) + ', max_features=' + str(max_features) +
          ', min_samples_leaf=' + str(min_samples_leaf) + ', min_samples_split='
          + str(min_samples_split) + ', max_depth=' + str(max_depth))

    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth,
                                                              random_state=42, verbose=1,
                                                              max_features=max_features,
                                                              min_samples_split=min_samples_split,
                                                              min_samples_leaf=min_samples_leaf))

    # We fit the model deleting variables that must not be included to do not have endogeneity (for example FRAUD
    # variable)
    regr_multirf.fit(X_train.drop(not_consider, axis=1), y_train)

    # We get R2 to determine how well is explaining the model
    score = regr_multirf.score(X_test.drop(not_consider, axis=1), y_test)
    print('R2 model ', score)

    # Now we bring the complete column dataframe with NaN row values
    df_nans = df.loc[~notnans].copy()
    df_not_nans = df.loc[notnans].copy()
    # Finally what we have to do is to estimate the NaN columns from the previous dataframe. For that we use
    # multioutput regression. This will estimate each specific column using Random Forest model. Basically we
    # need to predict dataframe column NaN values for each row in function of dataframe column not NaN values.
    df_nans[jcols] = regr_multirf.predict(df_nans[icols].drop(not_consider, axis=1))

    df_without_nans = pd.concat([df_nans, df_not_nans], axis=0, ignore_index=True)

    df = pd.merge(df, df_without_nans, how='left', on=on, suffixes=('', '_y'))

    for i in jcols:
        df[i] = df[i].fillna(df[i + '_y'])
        del df[i + '_y']

    filter_col = [col for col in df if col.endswith('_y')]
    for i in filter_col:
        del df[i]

    return df


def robust_scale(df:pd.DataFrame, keys:list, quantile_range=(25.0, 75.0)):
    """
            Scale features using statistics that are robust to outliers.
            This Scaler removes the median and scales the data according to the quantile range 
            (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) 
            and the 3rd quartile (75th quantile).
            :return: scaled df
            """
    from sklearn.preprocessing import RobustScaler

    robust_scaler = RobustScaler(quantile_range=quantile_range)

    df_cols = df.columns.values.tolist()
    df_cols.remove(keys)

    for i in df_cols:
        X = df[[i]]
        df[i] = robust_scaler.fit_transform(X)

    return df
