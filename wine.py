import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Any results you write to the current directory are saved as output.
data = load_wine()
X = data.data  # matriz contendo os atributos
y = data.target  # vetor contendo a classe (0 para maligno e 1 para benigno) de cada instância
feature_names = data.feature_names  # nome de cada atributo
target_names = data.target_names  # nome de cada classe
print(type(X))
print(f"Dimensões de X: {X.shape}\n")
print(f"Dimensões de y: {y.shape}\n")
print(f"Nomes dos atributos: {feature_names}\n")
print(f"Nomes das classes: {target_names}")

X_normalized = preprocessing.normalize(X, norm='l2')

# Lembrar de testar diferentes parâmetros no treinamento para explicar no relatório os melhores
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
# print(type(train_X[0]))
model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
# print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))

model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
# print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

gnb = GaussianNB()
y_pred = gnb.fit(train_X, train_y).predict(test_X)
print(test_X)
print(test_X.shape[0])
print("Number of mislabeled points out of a total %d points : %d"% (test_X.shape[0], (test_y != y_pred).sum()))


def k_fold(k):
    X_zero = X[0:59]
    X_one = X[59:130]
    X_two = X[130:]
    fold_length = round(len(X)/k)
    # print("fold length: ", fold_length)
    class_zero_proportion = len(X_zero)/len(X)
    class_one_proportion = len(X_one)/len(X)
    class_two_proportion = len(X_two)/len(X)
    # print('class zero proportion:', class_one_proportion)
    proportion_to_zero_class = round(class_zero_proportion * fold_length)
    proportion_to_one_class = round(class_one_proportion * fold_length)
    proportion_to_two_class = round(class_two_proportion * fold_length)

    # print('proportion_to_one_class:', proportion_to_one_class)
    lista = []
    temp_list = []

    for i in range(k):
        # print(round(class_zero_proportion * fold_length))
        # if(i == k-1):
        #   print('pegar o resto')
        #   temp_list.extend(X_zero[:proportion_to_zero_class])
        #   temp_list.extend(X_zero[:proportion_to_zero_class])
        #   temp_list.extend(X_zero[:proportion_to_zero_class])

        temp_list.extend(X_zero[:proportion_to_zero_class])
        X_zero = X_zero[proportion_to_zero_class:]
        temp_list.extend(X_one[:proportion_to_one_class])
        X_one = X_one[proportion_to_one_class:]
        temp_list.extend(X_two[:proportion_to_two_class])
        X_two = X_two[proportion_to_two_class:]
        # print('len(temp_list) =', len(temp_list))
        # print('len(X_zero):', len(X_one))
        lista.append(temp_list)

# k_fold(3)
