from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
import pylab as plot
import numpy
import pandas as pd
import csv
import lightgbm as lgbx
_author_ = 'Sebastian Palacio'

class gradient_boosting:
    def gradient_boosting(X,Y,names, output_pred, min_samples_split = 2000, min_samples_leaf = 200):

        fileNames = numpy.array(names)
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state= 0)
        min_sample_leaf = round((len(xTrain)) * 0.01)
        min_samples_split = min_sample_leaf * 10
        nEst = 500 #Número de trees
        depth = None  #En este modelo es irrelevante para la performance, sirve para ver la complejidad de las interacciones
        learnRate = 0.01 #Sirve para actualizar los residuos de las nuevas predicciones (se entrena el error del modelo anterior)
        subSamp = 1 #Al entrenar cada árbol individual en un subsample (0.5 recomienda Friedman) lo volvemos estocástico
        msError = []
        r2 = []
        predictions_list = []
        nTreeList = range(1, 1000, 1)
        for iTrees in nTreeList:
            fileModel = ensemble.GradientBoostingRegressor(min_samples_split = min_samples_split, min_samples_leaf= min_samples_leaf,
                                                       n_estimators= iTrees, max_depth= depth, learning_rate= learnRate,
                                                       loss= 'ls',verbose = 1, criterion='friedman_mse',
                                                           subsample=subSamp) #loss establece la función que evaluamos, en este caso el Least Square Summ de un OLS
            fileModel.fit(xTrain, yTrain)
            predictions = fileModel.predict(xTest)
            msError.append(mean_squared_error(yTest, predictions))
            r2.append(r2_score(yTest, predictions))




        #####Esta parte necesita stage prediction, porque va calculando por cada predicción

        print('MSE', min(msError))
        print('R2', max(r2))
        print('Indice MSE mínimo', msError.index(min(msError)))

        plot.figure()
        plot.plot(nTreeList, fileModel.train_score_, label = 'Training Set MSE')
        plot.plot(nTreeList, msError, label='Test Set MSE')
        plot.legend(loc = 'upper right')
        plot.xlabel('Number of Trees in Ensamble')
        plot.ylabel('MSE')
        plot.show()

        plot.figure()
        plot.plot(nTreeList, r2, label='Test R2')
        plot.legend(loc = 'upper right')
        plot.xlabel('Number of Trees in Ensamble')
        plot.ylabel('R2')
        plot.show()


        '''
        Often features do not contribute equally to predict the target response; in many situations the majority of the features are in fact irrelevant. When interpreting a model, the first question usually is: what are those important features and how do they contributing in predicting the target response?
        Individual decision trees intrinsically perform feature selection by selecting appropriate split points. This information can be used to measure the importance of each feature; the basic idea is: the more often a feature is used in the split points of a tree the more important that feature is. This notion of importance can be extended to decision tree ensembles by simply averaging the feature importance of each tree (see Feature importance evaluation for more details).
        '''
        featureImportance = fileModel.feature_importances_
        print('featureImportance', featureImportance)
        featureImportance = featureImportance / featureImportance.max()
        print('featureImportance Scaled', featureImportance)
        idxSorted = numpy.argsort(featureImportance)
        barPos = numpy.arange(idxSorted.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[idxSorted], align = 'center')
        plot.yticks(barPos, fileNames[idxSorted])
        plot.xlabel('Variable Importance')
        plot.subplots_adjust(left = 0.2, right = 0.9, top = 0.9, bottom = 0.1)
        plot.show()


    def gradient_boosting_predicted_fare(X, Y, names, split, X_0, X_1, X_2, X_3):
        fileNames = numpy.array(names)
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=531)

        nEst = 200  # Número de trees
        depth = 30  # En este modelo es irrelevante para la performance, sirve para ver la complejidad de las interacciones
        learnRate = 0.01  # Sirve para actualizar los residuos de las nuevas predicciones (se entrena el error del modelo anterior)
        subSamp = 0.5  # Al entrenar cada árbol individual en un subsample (0.5 recomienda Friedman) lo volvemos estocástico
        min_samples_split = split # Entre un 0.5% y un 1% de la data
        min_samples_leaf = int(0.1*split)

        fileModel = ensemble.GradientBoostingRegressor(min_samples_split = min_samples_split, min_samples_leaf= min_samples_leaf,
                                                       n_estimators=nEst, max_depth=depth, learning_rate=learnRate,
                                                       subsample=subSamp, loss='ls', random_state=0, verbose=1,
                                                       criterion='friedman_mse',
                                                       presort=True)  # loss establece la función que evaluamos, en este caso el Least Square Summ de un OLS
        fileModel.fit(xTrain, yTrain)
        '''
        predictions_0 = fileModel.predict(X_0)
        prediction = pd.DataFrame(predictions_0, columns=['predictions']).to_csv('df_p0.csv')


        predictions_1= fileModel.predict(X_1)
        prediction = pd.DataFrame(predictions_1, columns=['predictions']).to_csv('df_p1.csv')
        '''
        predictions_2 = fileModel.predict(X_2)
        prediction = pd.DataFrame(predictions_2, columns=['predictions']).to_csv('df_p2_holiday.csv')

        predictions_3 = fileModel.predict(X_3)
        prediction = pd.DataFrame(predictions_3, columns=['predictions']).to_csv('df_p3_holiday.csv')





