from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
import pylab as plot
import numpy

_author_ = 'Sebastian Palacio'


class random_forest:
    def random_forest(X,Y, names):

        xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.3,random_state=531) #Esto divide la muestra automáticamente en un 30% fijand
        fileNames = numpy.array(names)
        mseOos = []
        r2 = []
        nTreeList = range(1, 300, 1)
        for iTrees in nTreeList:
            print(iTrees)
            depth = None
            min_sample_leaf = round((len(xTrain.index)) * 0.01)
            min_sample_split = min_sample_leaf * 10
            maxFeat = 6 #A diferencia del bagging el random tree tambien aleatoriza sobre los atributos (Se elige 1/3 de la cantidad de atributos que itera 38/3 =
            fileModel = ensemble.RandomForestRegressor(criterion='mse', bootstrap=False,min_samples_leaf=min_sample_leaf,min_samples_split=min_sample_split,n_estimators=iTrees, max_depth=depth, max_features= maxFeat, oob_score=False, random_state= 531)
            fileModel.fit(xTrain,yTrain)
            prediction = fileModel.predict(xTest)
            mseOos.append(mean_squared_error(yTest,prediction))
            r2.append(r2_score(yTest,prediction))
        print('MSE', min(mseOos))
        print('R2 ', max(r2))

        plot.plot(nTreeList, r2)
        plot.xlabel('Number of Trees in Ensemble')
        plot.ylabel('R2')
        plot.ylim([0.0, 1.1*max(r2)])
        plot.show()

        plot.plot(nTreeList, mseOos)
        plot.xlabel('Number of Trees in Ensemble')
        plot.ylabel('MSE')
        plot.ylim([0.0, 1.1*max(mseOos)])
        plot.show()

        featureImportance = fileModel.feature_importances_

        featureImportance = featureImportance/featureImportance.max()
        sorted_idx = numpy.argsort(featureImportance)
        barPos = numpy.arange(sorted_idx.shape[0]) + 0.5
        plot.barh(barPos, featureImportance[sorted_idx], align = 'center')
        plot.yticks(barPos, fileNames[sorted_idx])
        plot.xlabel('Variable Importance')
        plot.show()