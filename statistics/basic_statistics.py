import os
import statsmodels.stats.stattools
import statsmodels.tsa.stattools
from pandas import DataFrame
from sklearn.metrics.regression import r2_score
import STRING
import matplotlib.pyplot as plot
from sklearn import linear_model
from mpmath import sqrt
import numpy as np
from sklearn.metrics import r2_score



_author_ = 'Sebastian Palacio'

class basic_statistics:


    def summ_file(file):
        data_describe = file.describe()
        print(data_describe)

    def corr_mat(file):
        corMat = DataFrame(file.iloc[:, :].corr())
        plot.pcolor(corMat)
        plot.show()
        corMat.to_csv('results\matrix_corr.csv', sep=';', encoding='utf-8')

    def paralell_plot(file):
        # Paralell coordinate plot
        summary = file.describe()
        nrows = len(file.index)
        colTravel = len(summary.columns)
        meanTravel = summary.iloc[1, colTravel - 1]
        sdTravel = summary.iloc[2, colTravel - 1]
        nDataCol = len(file.columns) - 1

        for i in range(nDataCol):
            dataRow = file.iloc[i, 1:nDataCol]
            normTarget = (file.iloc[i, nDataCol] - meanTravel) / sdTravel
            norm = plot.Normalize()
            dataRow.plot(color=plot.cm.jet(norm(normTarget)), alpha=0.5)
        plot.xlabel('Indice de Atributos')
        plot.ylabel('Valor de Atributos')
        plot.show()

    def errores_summ(X,Y):
        # Errores
        error = []
        fileModel = linear_model.LinearRegression()
        fileModel.fit(X,Y)
        prediction = fileModel.predict(X)
        r2 = r2_score(Y, prediction)
        print('R2 OLS ', r2)

        for i in range(len(Y)):
            error.append(Y[i]-prediction[i])

        # Squared Errors and Absolute value errors
        squaredError = []
        absError = []

        for val in error:
            squaredError.append(val * val)
            absError.append(abs(val))


        # MSE
        mse = sum(squaredError) / len(squaredError)
        print('MSE = ', mse)

        # RMSE = sqrt(MSE)
        #rmse = sqrt(mse)
        #print('RMSE = ', rmse)

        # MAE
        mae = sum(absError) / len(absError)
        print('MAE = ', mae)

        # compare MSE to target variance
        targetDeviation = []
        targetMean = sum(Y) / len(Y)
        for val in Y:
            targetDeviation.append((val - targetMean) * (val - targetMean))

        targetVariance = sum(targetDeviation) / len(targetDeviation)
        targetSD = sqrt(targetVariance)
        print('Target Variance = ', targetVariance)
        print('Target Standard Deviation = ', targetSD)



    def boxplot_var(file, var_out, var_condition):
        file.boxplot(var_out, by=var_condition)
        plot.xlabel('Attribute Index')
        plot.ylabel('Quartile Ranges')
        plot.show()


    def plot_var(var_x, var_y):
        plot.scatter(var_x, var_y, alpha=0.5, s=120)
        plot.xlabel('X Values')
        plot.ylabel('Y Values')
        plot.show()

    def dw_test(X,Y):
        fileModel = linear_model.LinearRegression()
        fileModel.fit(X,Y)
        prediction = fileModel.predict(X)
        error = []
        for i in range(len(Y)):
            error.append(Y[i] - prediction[i])
        print('dw test', statsmodels.stats.stattools.durbin_watson(error, axis=0))
        plot.scatter(range(len(Y)), error, alpha=0.5, s=120)
        plot.xlabel('X Values')
        plot.ylabel('Y Values')
        plot.show()

        '''
        The test statistic is approximately equal to 2*(1-r) where r is the sample autocorrelation of the residuals.
        Thus, for r == 0, indicating no serial correlation, the test statistic equals 2. This statistic will always be between 0
        and 4. The closer to 0 the statistic, the more evidence for positive serial correlation. The closer to 4,
        the more evidence for negative serial correlation.
        '''

    def ljung_box_test(X,Y):
        fileModel = linear_model.LinearRegression()
        fileModel.fit(X,Y)
        prediction = fileModel.predict(X)
        error = []
        for i in range(len(Y)):
            error.append(Y[i] - prediction[i])
        nobs = len(X[0])
        print('Ljung-Box Test', statsmodels.tsa.stattools.q_stat(error,nobs,type='ljungbox'))






