from utils import mnist_reader
from Functions import *

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

train_sizes, cv = Parameters(6, 100, 0.2)

BestEstimatorCurve(x_train, y_train, y_train, y_test, train_sizes, cv, estimator='decision tree')

BestEstimatorCurve(x_train, y_train, y_train, y_test, train_sizes, cv, estimator='perceptron')

PlotDecisionTree(cv, x_train, y_train, y_train, y_test, TrainS=train_sizes, criterio='entropy', minSamplesLeaf=5, minSamplesSplit=2, maxDepth=20)
PlotDecisionTree(cv, x_train, y_train, y_train, y_test, TrainS=train_sizes, criterio='entropy', minSamplesLeaf=5, minSamplesSplit=2, maxDepth=5)

PlotPerceptron(cv, x_train, y_train, y_train, y_test, TrainS=train_sizes, maxIter=50)
PlotPerceptron(cv, x_train, y_train, y_train, y_test, TrainS=train_sizes, maxIter=5)
PlotPerceptron(cv, x_train, y_train, y_train, y_test, TrainS=train_sizes, maxIter=2)

#DecisionTreeGridSearch(cv, x_train, y_train)

#PerceptronGridSearch(cv, x_train, y_train)

plt.show()
