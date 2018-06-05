from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from matplotlib import pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split

def variance_threshold(self: pd.DataFrame, key_variables:list, threshold=0.0):
    """        
    VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance
    doesn’t meet some threshold. By default, it removes all zero-variance features, i.e.
    features that have the same value in all samples.
    As an example, suppose that we have a dataset with boolean features,
    and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples.
    """
    column_names = self.columns.values.tolist()
    removed_var = []
    for i in key_variables:
        try:
            column_names.remove(i)
            removed_var.append(i)
        except:
            pass

    append_names = []
    for i in column_names:
        self_i = self[[i]]
        self_i = self_i.apply(pd.to_numeric, errors='coerce')
        self_i = self_i.dropna(how='any', axis=0)
        selection = VarianceThreshold(threshold=threshold)
        try:
            selection.fit(self_i)
            features = selection.get_support(indices=True)
            features = self_i.columns[features]
            features = [column for column in self_i[features]]
            selection = pd.DataFrame(selection.transform(self_i), index=self_i.index)
            selection.columns = features
            append_names.append(selection.columns.values.tolist())
        except:
            pass

    append_names = [item for sublist in append_names for item in sublist]
    append_names = list(set(append_names))
    self = self[removed_var + append_names]
    return self


def pca_reduction(df: pd.DataFrame, show_plot=False, variance=95.00, normalization=False):
    """
            This automatically calcualte a PCA to df taking into account the 95% of the dataset explained variance
            :param show_plot: Threshold Variance Plot
            :param variance: Dataset variance limit to consider in the PCA.
            :return: PCA df
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale

    siniestro_df = df[['id_siniestro', 'TEST']]
    del df['id_siniestro']
    del df['TEST']
    columns = len(df.columns)
    if normalization == True:
        X = scale(df)
    else:
        X = df
    pca = PCA(whiten=True, svd_solver='randomized', n_components=columns)

    pca.fit(X)
    cumsum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
    cumsum = list(cumsum)
    var = [value for value in cumsum if value <= variance]
    pca_components = len(var)

    if show_plot == True:
        plot.plot(cumsum)
        plot.title('Explained Variance Ratio (PCA)')
        plot.xlabel('N features')
        plot.axvline(x=pca_components, color='r', ls='--')
        plot.ylabel('Dataset Variance', rotation=90)
        import datetime
        import os
        DAY = datetime.datetime.today().strftime('%Y-%m-%d')
        path_probabilidad_day = 'final_files\\' + str(DAY) + '\\'
        os.makedirs(os.path.dirname(path_probabilidad_day), exist_ok=True)
        plot.savefig(path_probabilidad_day + 'pca.png')
        plot.close()

    print('PCA Components ', pca_components)

    pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized')
    if normalization == True:
        df = scale(df)
    else:
        pass
        pca.fit(df)
    df = pca.fit_transform(df)
    df = pd.DataFrame(df)

    df = pd.concat([df, siniestro_df], axis=1)

    return df

def robust_scale(df:pd.DataFrame, key_vars:list, quantile_range=(25.0, 75.0) ):
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
    for i in key_vars:
        df_cols.remove(i)

    for i in df_cols:
        X = df[[i]]
        df[i] = robust_scaler.fit_transform(X)

    return df



def training_test_valid(x: pd.DataFrame, y:pd.DataFrame):
    """
            Separate between training, test and valid using the next proportions:
            Training 70%
            Test 15%
            Valid 15%
            Also it keeps the same proportion between Fraud class inside Test an Valid.
            However, it excludes every fraud claim in the Train Set.
            """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, shuffle=True)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=.5, shuffle=True)

    train = pd.concat([x_train, y_train], axis=1)
    valid = pd.concat([x_valid, y_valid], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print('Train Shape ', train.shape)
    print('Valid Shape ', valid.shape)
    print('Test Shape ', test.shape)

    return train, valid, test


def training_test_valid_by_date(x: pd.DataFrame, y: pd.DataFrame,  label='PASSENGER_SUM_DAY',
                                test_date='RATE_Apr16'):
    """
            Separate between training, test and valid using the next proportions:
            Training 70%
            Test 15%
            Valid 15%
            Also it keeps the same proportion between Fraud class inside Test an Valid.
            However, it excludes every fraud claim in the Train Set.
            """
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    dataset = pd.concat([x, y], axis=1)
    test_set = dataset[dataset[test_date] == dataset[test_date].max()]
    train_set = dataset[dataset[test_date] != dataset[test_date].max()]

    del test_set[test_date]
    del train_set[test_date]

    test_set = test_set.reset_index(drop=True)
    train_set = train_set.reset_index(drop=True)

    print('Train Shape ', train_set.shape)
    print('Test Shape ', test_set.shape)


    return train_set, test_set