import numpy
from sklearn import linear_model
from mpmath import sqrt
import matplotlib.pyplot as plot
from sklearn.metrics import r2_score



_author_ = 'Sebastian Palacio'

class ridge_regression:

    def ridge_regression(xList, target):

        ###OTRO MÉTODO PARA CONTROLAR OVERFITTING: RIDGE REGRESSION (CONTROLA MEDIANTE PENALIZACIÓN DE LOS COEF DE REGRESIÓN pp.110)
        '''
        Este método es una introducción a los modelos de regresiones lineales penalizadas. El OLS busca un escalar beta_0 y un vector beta que satisface:

        beta_0*,beta* = argmin_{beta_0,beta}( Sum (y_i -(beta_0 + x_i beta))^2  )*1/m    (1)

        En definitiva, los coeficientes beta_0* y beta* son los coeficientes de la solución OLS. El step-forward regression ajusta hacia atrás limitando
        el número de atributos utilizados. Esto equivale a poner una restricción de que alguna de las entradas en el vector beta sean cero. La aproximación
        mediante 'coefficient penalized regression' logra lo mismo, pero haciendo a todos los coeficientes pequeños en vez de hacer cero a algunos. Una
        versión de este tipo de aproximación es la 'ridge regression':

        beta_0*,beta* = argmin_{beta_0,beta}[( Sum (y_i -(beta_0 + x_i beta))^2  )*1/m +alpha*beta^T*beta ]   (2)

        La diferencia con (1) es que se le adiciona el término alpha*beta^T*beta que es el cuadrado de la normal Euclideana de beta (el vector de
        coeficientes). Antes, el número de atributos era el 'parámetro de complejidad' que variabamos. Ahora es beta es el parámetro de complejidad.
        Si alpha = 0, el problema es un OLS. Cuando aumenta alpha, beta se aproxima a cero, y solo beta_0 (la constante) puede predecir y.

        '''
        indices = range(len(xList))

        xListTest = [xList[i] for i in indices if i % 3 == 0]
        xListTrain = [xList[i] for i in indices if i % 3 != 0]
        targetTest = [target[i] for i in indices if i % 3 == 0]
        targetTrain = [target[i] for i in indices if i % 3 != 0]


        xTrain = numpy.array(xListTrain)
        xTest = numpy.array(xListTest)
        yTrain = numpy.array(targetTrain)
        yTest = numpy.array(targetTest)

        alphaList = [0.1 ** i for i in[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # crea una lista a partir de la función 0.1^i con los valores (0-6)'''
        rmsError = []
        maeError = []
        betaCoef = []

        for alph in alphaList:
            fileRidgeModel = linear_model.Ridge(alpha=alph)
            fileRidgeModel.fit(xTrain, yTrain)
            rmsError.append(numpy.linalg.norm((yTest - fileRidgeModel.predict(xTest)), 2)*numpy.linalg.norm((yTest - fileRidgeModel.predict(xTest)), 2) / len(yTest) )

            absVal = numpy.linalg.norm((yTest - fileRidgeModel.predict(xTest)), 1) / len(yTest)
            betas = fileRidgeModel.coef_
            maeError.append(absVal)
            betaCoef.append(betas)

        # Ridge(alpha) This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
        '''
        alpha : {float, array-like}, shape (n_targets)
        Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization. Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.
        If an array is passed, penalties are assumed to be specific to the targets. Hence they must correspond in number.
        '''

        print('RMS Error - alpha')
        for i in range(len(rmsError)):
            print('MSE', rmsError[i], alphaList[i])
            print('MAE',maeError[i], alphaList[i])
            print('BETAS',betaCoef[i], alphaList[i])



        '''
        En el stepwise regression teníamos una secuencia de modelos diferentes (en función de la cantidad de atributos), aquí también pero en función
        de distintos alphas (el parámetro que determina la severidad en la penalidad de beta).
        '''

        # PLOT CURVE OF OUT-SAMPLE ERROR VERSUS ALPHA (BEFORE WAS VERSUS N ATTRIBUTTES)
        x = range(len(rmsError))
        plot.plot(x, rmsError, 'k')
        plot.xlabel('-log(alpha)')
        plot.ylabel('MSE')
        plot.title('Ridge Regression')
        plot.show()

        '''
        Observamos los errores como función del parámetro de complejidad alpha. A la izquierda se muestran los mayores alpha (donde -log(alpha)= 0,
        alpha es igual a 1). Notamos que los errores son crecientes con menor alpha (cuando alhpa tiende a cero nos acercamos al modelo de
        stepwise regression). Es decir que favorece la existencia de la restricción impuesta por la Ridge Regression.
        '''

        # PLOT HISTOGRAM OF OUT SAMPLE ERRORS FOR BEST ALPHA VALUE (BEFORE WAS NUMBER OF ATTRIBUTES)

        indexBest = rmsError.index(min(rmsError))
        print('indexBest', indexBest)
        alph = alphaList[indexBest]
        print('alph', alph)
        '''
        fileRidgeModel = linear_model.Ridge(alpha=alph)
        fileRidgeModel.fit(xTrain, yTrain)
        errorVector = yTest - fileRidgeModel.predict(xTest)
        plot.hist(errorVector)
        plot.xlabel('Bind Boundaries')
        plot.ylabel('Counts')
        plot.show()
        '''

        # Scatter Plot of actual versus predicted
        plot.scatter(fileRidgeModel.predict(xTest), yTest, s=100, alpha=0.25)
        plot.xlabel('Predicted Values')
        plot.ylabel('Real Values')
        plot.title('Ridge Regression')
        plot.show()

        r2_scores = r2_score(yTest, fileRidgeModel.predict(xTest))
        print('r2 Ridge', r2_scores)
