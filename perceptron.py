import sys 
from sklearn import datasets
import seaborn.apionly as sns
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

iris = sns.load_dataset('iris')

#aqui está o código para classificar as espécies setosa e versicolor, a acurácia foi de 100% para as duas classes 
mask = iris['species'].isin(['virginica'])
iris1=iris[~mask]
iris1=iris1.sample(frac=1).reset_index(drop=True)

dataframe_train=iris1[0:80]
dataframe_test=iris1[-20:]


x_train = dataframe_train.drop('species', 1)
y_train = dataframe_train['species']


x_test = dataframe_test.drop('species', 1)
y_test = dataframe_test['species']


clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
print(clf.score(x_test,y_test))
#------------------------------------------------------------------------------------------------------------------
#aqui está o código para classificar as espécies setosa, versicolor e virginica, a acurácia foi de 78% para as três classes 
iris2=iris
iris2=iris2.sample(frac=1).reset_index(drop=True)

dataframe_train=iris2[0:100]
dataframe_test=iris2[-50:]


x_train = dataframe_train.drop('species', 1)
y_train = dataframe_train['species']


x_test = dataframe_test.drop('species', 1)
y_test = dataframe_test['species']

#print(dataframe_train)
#print(dataframe_test)

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
print(clf.score(x_test,y_test))
#------------------------------------------------------------------------------------------------------------------------

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10, random_state=21,tol=0.000000001)
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_test)

print(clf.score(x_test,y_test))
print(y_test)
print(y_pred)


