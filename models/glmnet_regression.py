import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
import numpy as np
from sklearn import linear_model
import os
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

_author_ = 'Sebastian Palacio'


class glmnet_regression:


    def enet_reg_alpha(X, Y):
        #PASO 1: OUT-OF-SAMPLE PERFORMANCE

        fileModel = ElasticNetCV(cv=3).fit(X, Y)


        # Display Results
        plot.figure()
        plot.plot(fileModel.alphas_, fileModel.mse_path_, ':')  # este plotea los alphas y los mse
        plot.plot(fileModel.alphas_, fileModel.mse_path_.mean(axis=-1), label='Average MSE Across Folds',
                  linewidth=2)  # Este le da el formato y el label al ploteo de MSE
        plot.axvline(fileModel.alpha_, linestyle='--', label='CV Estimate of Best Alpha')
        plot.semilogx()
        plot.legend()
        ax = plot.gca()
        ax.invert_xaxis()
        plot.xlabel('alpha')
        plot.ylabel('Mean Square Errror')
        plot.axis('tight')
        plot.show()

        alphaStar = fileModel.alpha_
        print('alpha that Min CV Error ', alphaStar)
        print('Minimum MSE ', min(fileModel.mse_path_.mean(axis=-1)))


    def elastic_reg_betas(X, Y,names, alphaStar):

        def S(z, gamma):
            if gamma >= abs(z):
                return 0.0
            return (z / abs(z)) * (abs(z) - gamma)


        nrows = len(X)
        ncols = len(X[0])

        # DETERMINE VALOR OF LAMBDA THAT MAKE BETA = 0

        xy = [0.0] * ncols

        for i in range(nrows):
            for j in range(ncols):
                xy[j] += X[i][j] * Y[i]  # Esto hace X*Y para ver la correlacion entre atributo y target

        maxXY = 0.0
        for j in range(ncols):  # Aquí tomo el de mayor correlación
            val = abs(xy[j]) / nrows
            if val > maxXY:
                maxXY = val

        lam = maxXY / alphaStar  # Este ES LAMBDA QUE CORRESPONDE A BETA = 0
        print('lambda_max ', lam)

        # INITIALIZATE BETA
        beta = [0.0] * ncols

        # INITIALIZATE MATRIX OF BETA
        betaMat = []
        lambdaMat = []
        mse = []
        betaMat.append(list(beta))

        # COMENZAMOS A REDUCIR LAMBDA
        nSteps = 100
        lamMult = 0.93  # Es el sugerido por los autores (100 steps gives reduction by factor of 1000 in lambda)
        residualList = []
        nzList = []

        for iStep in range(nSteps):
            lam = lam * lamMult

            deltaBeta = 100
            eps = 0.01
            iterStep = 0
            betaInner = list(beta)
            while deltaBeta > eps:
                iterStep += 1
                if iterStep > 100: break

                # ciclo a través de los atributos y actualizando registro por registro
                betaStart = list(betaInner)
                residualList = []
                for iCol in range(ncols):
                    xyj = 0
                    for i in range(nrows):
                        labelHat = sum([X[i][j] * betaInner[j] for j in range(ncols)])
                        residual = Y[i] - labelHat
                        residual2 = residual * residual
                        residualList.append(residual2)
                        xyj += X[i][iCol] * residual
                    mse = sum(residualList) / len(Y)
                    uncBeta = xyj / nrows + betaInner[iCol]
                    betaInner[iCol] = S(uncBeta, lam * alphaStar) / (1 + lam * (1 - alphaStar))

                sumDiff = sum([abs(betaInner[n] - betaStart[n]) for n in range(ncols)])
                sumBeta = sum([abs(betaInner[n]) for n in range(ncols)])

                deltaBeta = sumDiff / sumBeta
            print(iStep, iterStep, lam, mse)
            beta = betaInner  # Este devuelve la estimación de beta para cada paso en el que se va reduciendo el error
            betaMat.append(beta)  # Agrega las estimaciones de beta

            nzBeta = [index for index in range(ncols) if
                      beta != 0.0]  # Armamos un indice para los beta que son significativos
            for q in nzBeta:  # Evaluamos si el indice que armamos no está en nzList, lo agregamos
                if (q in nzList) == False:
                    nzList.append(q)

        # PRINT OUT THE ORDER LIST OF BETAS
        nameList = [names[nzList[i]] for i in range(len(nzList))]
        print(nameList)

        nPts = len(betaMat)
        for i in range(ncols):
            coefCurve = [betaMat[k][i] for k in range(nPts)]
            xaxis = range(nPts)
            plot.plot(xaxis, coefCurve)

        plot.xlabel('Steps Taken')
        plot.ylabel('Coefficient Values')
        plot.show()




    def enet_r2(X,Y, alphaStar,lambdaStar):
        enet = ElasticNet(alpha=alphaStar, l1_ratio=lambdaStar, fit_intercept=False, normalize=False)
        indices = range(len(X))
        xTest = [X[i] for i in indices if i % 3 == 0]
        xTrain = [X[i] for i in indices if i % 3 != 0]
        targetTest = [Y[i] for i in indices if i % 3 == 0]
        targetTrain = [Y[i] for i in indices if i % 3 != 0]
        y_pred_enet = enet.fit(xTrain, targetTrain).predict(xTest)
        r2_score_enet = r2_score(targetTest, y_pred_enet)
        mse = mean_squared_error(targetTest, y_pred_enet)

        print('mse', mse)
        print("r^2 on test data : %f" % r2_score_enet)


