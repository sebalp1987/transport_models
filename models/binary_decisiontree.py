from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot
import numpy
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error

_author_ = 'Sebastian Palacio'

class binary_decisiontree:

    def binary_decision_tree_depth(X, Y):

        nrow = len(X[0])
        x = [[s] for s in X]

        depthList = range(1,30,1)
        xvalMse = []
        nxval = 10
        r2 = []
        for iDepth in depthList:
            for ixval in range(nxval):
                idxTest = [a for a in range(nrow) if a%nxval == ixval%nxval]
                idxTrain = [a for a in range(nrow) if a % nxval != ixval % nxval]

                xTrain = [X[r] for r in idxTrain]
                xTest = [X[r] for r in idxTest]
                yTrain = [Y[r] for r in idxTrain]
                yTest = [Y[r] for r in idxTest]

                fileModel = DecisionTreeRegressor(max_depth=iDepth, criterion='mse',splitter='best',min_samples_split=10, min_samples_leaf=10)
                fileModel.fit(xTrain,yTrain)

                treePrediction = fileModel.predict(xTest)
                error = [yTest[r] - treePrediction[r] for r in range(len(yTest))]
                r2_calc = r2_score(yTest,treePrediction)

                if ixval == 0:
                    oosErrors = sum([e*e for e in error])

                else:
                    oosErrors += sum([e*e for e in error])

            mse = oosErrors/nrow
            xvalMse.append(mse)
            r2.append(r2_calc)
        plot.plot(depthList, xvalMse)
        plot.axis('tight')
        plot.xlabel('Tree Depth')
        plot.ylabel('MSE')
        plot.show()

        plot.plot(depthList, r2)
        plot.axis('tight')
        plot.xlabel('Tree Depth')
        plot.ylabel('R2')
        plot.show()


        print('Min MSE', min(xvalMse))
        print('Indice Min MSE', xvalMse.index(min(xvalMse)))

        print('Max R2', max(r2))
        print('Indice Max R2', r2.index(max(r2)))


    def binary_decision_tree(X, Y, depth, names, plot_fig=False):

        fileNames = numpy.array(names)

        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=531)
        min_sample_leaf = round((len(xTrain.index)) * 0.02)
        min_sample_split = min_sample_leaf * 10
        fileModel = DecisionTreeRegressor(max_depth=depth, criterion='mse',splitter='best', min_samples_split=min_sample_split, min_samples_leaf=min_sample_leaf)
        fileModel.fit(xTrain,yTrain)
        prediction = fileModel.predict(xTest)
        r2_calc = r2_score(yTest, prediction)
        mse = mean_squared_error(yTest, prediction)

        '''
        with open('results\\binary_tree.png','w') as f:
            f = tree.export_graphviz(fileModel, out_file=f)
        '''
        if plot_fig == True:
            featureImportance = fileModel.feature_importances_

            rule = fileModel.decision_path(xTrain)

            featureImportance = featureImportance * 100 / featureImportance.max()
            print(featureImportance)
            sorted_idx = numpy.argsort(featureImportance)
            barPos = numpy.arange(sorted_idx.shape[0]) + 0.5
            plot.barh(barPos, featureImportance[sorted_idx], align='center')
            plot.yticks(barPos, fileNames[sorted_idx])
            plot.xlabel('Variable Importance')
            plot.show()
            print(featureImportance[sorted_idx]/100)
            print(fileNames[sorted_idx])

        '''
        plot.figure()
        plot.plot(X, Y, label = 'True Y')
        plot.plot(X, yHat, label='Tree Prediction', linestyle = '--')
        plot.legend(bbox_to_anchor = (1,0.2))
        plot.axis('tight')
        plot.xlabel('X')
        plot.ylabel('Y')
        plot.show()
        '''
        return mse

    def iter_mse(X,Y,depth):
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=531)
        iter = range(1,depth,1)
        mseList = []

        for i in iter:
            fileModel = DecisionTreeRegressor(max_depth=i, criterion='mse', splitter='best', min_samples_split=900,
                                              min_samples_leaf=90)
            fileModel.fit(xTrain, yTrain)
            yHat = fileModel.predict(xTest)
            mse = mean_squared_error(yTest,yHat)
            mseList.append(mse)

        minMSE = mseList.index(min(mseList))+1
        print(depth, mseList)
        print('Min MSE', min(mseList))
        print('Indice Min MSE', minMSE)

        plot.plot(iter,mseList)
        plot.axis('tight')
        plot.xlabel('Tree Depth')
        plot.ylabel('MSE')
        plot.show()




    def iter_r2(X,Y,depth):
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=531)
        iter = range(1,depth,1)
        r2List = []

        for i in iter:
            fileModel = DecisionTreeRegressor(max_depth=i, criterion='mse', splitter='best', min_samples_split=900,
                                              min_samples_leaf=90)
            fileModel.fit(xTrain, yTrain)
            yHat = fileModel.predict(xTest)
            r2 = r2_score(yTest,yHat)
            r2List.append(r2)

        inR2 = r2List.index(max(r2List))+1

        print(depth, r2List)
        print('Max R2', max(r2List))
        print('Indice Max R2', inR2)

        plot.plot(iter,r2List)
        plot.axis('tight')
        plot.xlabel('Tree Depth')
        plot.ylabel('R2')
        plot.show()


    def plot_surface(X,Y):
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=531)

        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(xTrain, yTrain)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        plot.show()
