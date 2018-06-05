from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import pylab as plot
import numpy
import random

_author_ = 'Sebastian Palacio'

class bagging:
    def bagging(X, Y, treeDepth, names):
        fileNames = numpy.array(names)
        nrows = len(X)
        ncols = len(X[0]) 

        nSample = int(nrows * 0.3)
        idxTest = random.sample(range(nrows),nSample)
        idxTest.sort()
        idxTrain = [idx for idx in range(nrows) if not(idx in idxTest)]

        xTrain = [X[r] for r in idxTrain]
        xTest = [X[r] for r in idxTest]
        yTrain = [Y[r] for r in idxTrain]
        yTest = [Y[r] for r in idxTest]

        numTreesMax = 100


        modelList = []
        predList = []
        featureList = []
        sortedFeature = []

        bagFract = 1 #Determina cuantas muestras se toman para hacer el bootstrap. El paper original recomienda tomar el mismo tamaño que el set original, por ello se deja en 1

        nBagSamples = int(len(xTrain) * bagFract)

        for iTrees in range(numTreesMax):
            print(iTrees)
            idxBag = []
            for i in range(nBagSamples):
                idxBag.append(random.choice(range(len(xTrain))))
            xTrainBag = [xTrain[i] for i in idxBag]
            yTrainBag = [yTrain[i] for i in idxBag]
            min_sample_leaf = round((len(xTrain)) * 0.02)
            min_sample_split = min_sample_leaf * 10
            modelList.append(DecisionTreeRegressor(max_depth=treeDepth, criterion='mse',splitter='best', min_samples_split=min_sample_split, min_samples_leaf=min_sample_leaf))
            modelList[-1].fit(xTrainBag, yTrainBag)
            featureImportance = modelList[-1].feature_importances_
            print('fea',featureImportance)
            sorted_idx = numpy.argsort(featureImportance)
            print('idx',sorted_idx)
            featureList.append(featureImportance)
            sortedFeature.append(sorted_idx)

            barPos = numpy.arange(sorted_idx.shape[0]) + 0.5
            plot.barh(barPos, featureImportance[sorted_idx], align='center')
            plot.yticks(barPos, fileNames[sorted_idx])
            plot.xlabel('Variable Importance')
            plot.show()

            latestPrediction = modelList[-1].predict(xTest)
            predList.append(list(latestPrediction))


        mse = []
        allPredictions = []
        r2 = []

        for iModels in range(len(modelList)):
            prediction = []
            for iPred in range(len(xTest)):
                prediction.append(sum([predList[i][iPred] for i in range(iModels+1)])/(iModels+1))

            allPredictions.append(prediction)
            errors = [yTest[i] - prediction[i] for i in range(len(yTest))]
            r2_calcu = r2_score(yTest, prediction)
            mse.append(sum([e*e for e in errors])/len(yTest))
            r2.append(r2_calcu)

        nModels = [i + 1 for i in range(len(modelList))]

        plot.plot(nModels, mse)
        plot.axis('tight')
        plot.xlabel('Number of Models in Ensemble')
        plot.ylabel('Mean Squared Error (MSE)')
        plot.show()

        plot.plot(nModels, r2)
        plot.axis('tight')
        plot.xlabel('Number of Models in Ensemble')
        plot.ylabel('R2')
        plot.show()

        print('Min MSE', min(mse))
        print('Max R2', max(r2))


