from sklearn import ensemble
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

class extreme_randomize:
    def tunning(Train, Valid, scores='neg_mean_absolute_error', label='PASSENGERS', key='DATE'):

        Train = pd.concat([Train, Valid], axis=0, ignore_index=True)
        yTrain = Train[[label]]
        xTrain = Train

        min_sample_leaf = round((len(xTrain.index)) * 0.01)
        min_sample_split = min_sample_leaf * 10

        features = 'sqrt'

        tuned_parameters = [{'bootstrap': [True], 'min_samples_leaf': [min_sample_leaf],
                                 'min_samples_split': [min_sample_split], 'n_estimators': [200, 300, 500],
                                 'max_depth': [50, 100, 200],
                                 'max_features': [features],
                                 'oob_score': [True, False], 'random_state': [531], 'verbose': [1],
                                 'n_jobs': [-1]
                                 }]

        fileModel = GridSearchCV(ensemble.ExtraTreesRegressor(), param_grid=tuned_parameters, cv=10,
                                 scoring=scores)
        if key is not None:
            fileModel.fit(xTrain.drop([key], axis=1).values, yTrain[label].values)
        else:
            fileModel.fit(xTrain.values, yTrain[label].values)
        print("Best parameters set found on development set:")
        print()
        dict_values = fileModel.best_params_
        print(dict_values)
        print()
        print("Grid scores on development set:")
        print()
        means = fileModel.cv_results_['mean_test_score']
        stds = fileModel.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, fileModel.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        max_depth = int(dict_values['max_depth'])
        n_estimators = int(dict_values['n_estimators'])
        bootstrap = bool(dict_values['bootstrap'])
        oob_score = bool(dict_values['oob_score'])
        max_features = str(dict_values['max_features'])

        df = pd.DataFrame.from_dict(dict_values, orient="index")
        df.to_csv('ert_tunning.csv', sep=';', encoding='latin1', index=False)

        return max_depth, n_estimators, bootstrap, oob_score, max_features, min_sample_leaf, min_sample_split, max_features



    def evaluation(Train, Test, Valid, max_depth, n_estimators, bootstrap, oob_score, max_features, min_sample_leaf,
                   min_sample_split, label='PASSENGER', output_prob = False, key='DATE'):
        Train = pd.concat([Train, Valid], axis=0)
        max_features = 6
        min_sample_leaf = round((len(Train.index)) * 0.01)


        yTrain = Train[[label]]
        xTrain = Train
        del xTrain[label]
        yTest = Test[[label]]
        xTest = Test
        nTreeList = range(1, 300, 1)
        mseOos = []
        r2_list = []
        for iTrees in nTreeList:
            params = {'criterion': 'mse', 'bootstrap': bootstrap,
                      'min_samples_leaf': min_sample_leaf,
                      'min_samples_split': min_sample_split,
                      'n_estimators': iTrees,
                      'max_depth': max_depth, 'max_features': max_features,
                      'oob_score': oob_score,
                      'random_state': 531, 'verbose': 1,
                      'n_jobs': 1}
            if key is not None:

                names = xTrain.drop([key], axis=1).columns.values
                fileNames = np.array(names)
                fileModel = ensemble.ExtraTreesRegressor(**params)

                fileModel.fit(xTrain.drop([key], axis=1).values, yTrain.values)

                # METRICS
                mse = mean_squared_error(yTest, fileModel.predict(xTest.drop([key, label], axis=1)))
                medabs = median_absolute_error(yTest, fileModel.predict(xTest.drop([key, label], axis=1)))
                r2 = r2_score(yTest, fileModel.predict(xTest.drop([key, label], axis=1)))
            else:
                names = xTrain.columns.values
                fileNames = np.array(names)
                fileModel = ensemble.ExtraTreesRegressor(**params)

                fileModel.fit(xTrain.values, yTrain.values)

                # METRICS
                mse = mean_squared_error(yTest, fileModel.predict(xTest.drop([label], axis=1)))
                medabs = median_absolute_error(yTest, fileModel.predict(xTest.drop([label], axis=1)))
                r2 = r2_score(yTest, fileModel.predict(xTest.drop([label], axis=1)))
            mseOos.append(mse)
            r2_list.append(r2)
        print('MSE ', mse)
        print('MEDIAN ABS DEVIATION ', medabs)
        print('R2 ', r2)

        # PLOT TRAINING DEVIANCE
        '''
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(fileModel.staged_predict(xTest)):
            test_score[i] = fileModel.loss_(yTest, y_pred)

        plot.figure(figsize=(12, 6))
        plot.subplot(1, 2, 1)
        plot.title('Deviance')
        plot.plot(np.arange(params['n_estimators']) + 1, fileModel.train_score_, 'b-',
                 label='Training Set Deviance')
        plot.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plot.legend(loc='upper right')
        plot.xlabel('Boosting Iterations')
        plot.ylabel('Deviance')

        '''
        plot.plot(nTreeList, r2_list)
        plot.xlabel('Number of Trees in Ensemble')
        plot.ylabel('R2')
        plot.ylim([0.0, 1.1 * max(r2_list)])
        plot.show()

        plot.plot(nTreeList, mseOos)
        plot.xlabel('Number of Trees in Ensemble')
        plot.ylabel('MSE')
        plot.ylim([0.0, 1.1 * max(mseOos)])
        plot.show()

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance / featureImportance.max()
        sorted_idx = numpy.argsort(featureImportance)
        barPos = numpy.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align='center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()

        if output_prob == True:
            prediction = fileModel.predict(xTest.drop([key, label], axis=1))
            if Valid is not None:
                prediction_valid = fileModel.predict(Valid.drop([label, key], axis=1).values)
                prediction_valid = pd.DataFrame(prediction_valid, columns=['prediction_valid'])
                prediction_valid = pd.concat([prediction_valid, Valid[[label]], Valid[[key]]], axis=1)
                prediction_valid.to_csv('probabilities_valid.csv', sep=';', encoding='latin1', index=False)


            prediction = pd.DataFrame(prediction, columns= ['predictions'])

            prediction = pd.concat([prediction, xTest[[key, 'SIZE']]], axis=1)
            prediction = pd.concat([prediction, yTest], axis=1)
            prediction.to_csv('probabilities.csv', sep=';', encoding='latin1', index=False)

            # El valid sirve como base para ver la variación estructural. Hay que sacarle la tendencia a lo que se predice usando este valid