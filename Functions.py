import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def Parameters(nSamples, nSplit, testSize):

    trainSizes=np.logspace(np.log10(0.05), np.log10(1.0), num=nSamples)
    cv=ShuffleSplit(n_splits=nSplit,test_size=testSize,random_state=0)
    return trainSizes, cv

def plot(train_sizes, train_scores, test_scores, invert_score=True, title='', ylim=None, description=''):

    plt.figure()
    plt.suptitle(description)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.grid(True)
    if invert_score:
        train_scores = 1-train_scores
        test_scores = 1-test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test error")
    plt.legend(loc="best")


def PlotDecisionTree(CrossVal, x, y, x_test, y_test, TrainS, criterio, minSamplesSplit=2, minSamplesLeaf=1, maxDepth=None):

    estimator = tree.DecisionTreeClassifier(criterion=criterio, max_depth=maxDepth,
                                            min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf)

    trainSizes, trainScore, testScore = learning_curve(estimator , x, y, train_sizes=TrainS, cv=CrossVal, n_jobs=-1)
    descr=str('DecisionTree usando %s con maxDepth: %d , minSampleSplit: %d , minSampleLeaf: %d ' % (criterio, maxDepth, minSamplesSplit, minSamplesLeaf))
    plot(trainSizes, trainScore, testScore, title='DecisionTree', description=descr)
    estimator.fit(x,y)
    print '%.2f%% score nel training del DecisionTree' % estimator.score
    y_pred = estimator.predict(x_test)
    accuracy= 100*accuracy_score(y_test, y_pred)
    matrix=confusion_matrix(y_test, y_pred)
    print '%.2f%% Accuratezza predizione per DecisionTree usando %s con maxDepth: %d , minSampleSplit: %d , minSampleLeaf: %d ' % (accuracy, criterio, maxDepth, minSamplesSplit, minSamplesLeaf)
    print 'matrice : \n', matrix
    print 'Classification report: \n', classification_report(y_test, y_pred)

def PlotPerceptron( CrossVal, x, y, x_test, y_test, TrainS, maxIter=5):

    estimator =Perceptron(max_iter=maxIter)

    trainSizes, trainScore, testScore = learning_curve(estimator, x, y, train_sizes=TrainS, cv=CrossVal, n_jobs=-1)
    descr=str('Perceptron usando maxIter: %d ' % maxIter)
    plot(trainSizes, trainScore, testScore, title='Perceptron', description=descr)
    clf = estimator.fit(x,y)
    y_pred=clf.predict(x_test)
    print '%.2f%% score nel training del DecisionTree' % clf.score
    accuracy= 100*accuracy_score(y_test, y_pred)
    matrix=confusion_matrix(y_test, y_pred)
    print '%.2f%% Accuratezza per perceptron usando maxIter: %d ' % (accuracy, maxIter)
    print 'matrice: \n', matrix
    print 'Classification report: \n', classification_report(y_test, y_pred)


def PerceptronGridSearch(cv, x_train, y_train):

    parameter_grid={'max_iter': range(1,35)}
    estimator = Perceptron()
    grid_search = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train,y_train)
    print "Best Score Perceptron: %.2f%%", 100*(grid_search.best_score_)
    print "Best params Perceptron: {}".format(grid_search.best_params_)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    print 'Accuracy scores on training set'
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

def DecisionTreeGridSearch(cv, x_train, y_train):

    parameter_grid = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,15)}
    estimator = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, min_samples_split=2)
    grid_search = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train,y_train)
    print "Best Score DecisionTree: %.2f%%", 100*(grid_search.best_score_)
    print "Best params DecisionTree: {}".format(grid_search.best_params_)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    print 'Accuracy scores on training set'
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def BestEstimatorCurve(x_train, y_train, x_test, y_test, train_sizes, cv, estimator=''):

    if estimator == 'decision tree':
        parameter_grid = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 15)}
        estimator = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, min_samples_split=2)
        clf = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1, scoring='accuracy')
        trainSizes, trainScore, testScore = learning_curve(clf , x_train, y_train, train_sizes=train_sizes, cv=cv, n_jobs=-1)
        plot(trainSizes, trainScore, testScore, title='DecisionTree')
        y_pred = clf.fit(x_train,y_train).predict(x_test)
        accuracy= 100*accuracy_score(y_test, y_pred)
        matrix=confusion_matrix(y_test, y_pred)
        print "Miglior punteggio DecisionTree nel training: %.2f%%", 100*(clf.best_score_)
        print '{}% Accuratezza predizione Decision Tree usando i parametri che restituiscono il miglior punteggio nel train : {}'.format(accuracy,clf.best_params_)
        print 'matrice : \n', matrix
        print 'Classification report: \n', classification_report(y_test, y_pred)

    elif estimator == 'perceptron':
        parameter_grid = {'max_iter':range(1,35)}
        estimator = Perceptron()
        clf = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1, scoring='accuracy')
        trainSizes, trainScore, testScore = learning_curve(clf , x_train, y_train, train_sizes=train_sizes, cv=cv, n_jobs=-1)
        plot(trainSizes, trainScore, testScore, title='Perceptron')
        y_pred= clf.fit(x_train,y_train).predict(x_test)
        accuracy= 100*accuracy_score(y_test, y_pred)
        matrix=confusion_matrix(y_test, y_pred)
        print "Miglior punteggio Perceptron nel training: %.2f%%", 100*(clf.best_score_)
        print '{}% Accuratezza predizione del Perceptron usando i parametri che restituiscono il miglior punteggio nel train : {}'.format(accuracy,clf.best_params_)
        print 'matrice : \n', matrix
        print 'Classification report: \n', classification_report(y_test, y_pred)
