import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('wine.csv')


def k_fold(k):
    """
        Gera k folds de um dataframe
        :param k: número de folds
        :return: lista: lista contendo dataframes sendo cada dataframe correspondente a um fold
    """
    class_one_df = df.loc[df['class'] == 1]
    class_two_df = df.loc[df['class'] == 2]
    class_three_df = df.loc[df['class'] == 3]
    fold_length = round(len(df)/k)
    proportion_to_one_class = round(len(class_one_df)/len(df) * fold_length)
    proportion_to_two_class = round(len(class_two_df)/len(df) * fold_length)
    proportion_to_three_class = round(len(class_three_df)/len(df) * fold_length)

    lista = []

    for i in range(k):
        temp_list = pd.DataFrame()

        # Se for última iteração pega tudo que sobrou, mesmo que fique fold maior
        if i == k:
            temp_list = pd.append([class_one_df])
            temp_list = pd.append([temp_list, class_two_df])
            temp_list = pd.append([temp_list, class_three_df])
            lista.append(temp_list)
        else:
            # depois testar pegar aleatório de class_one_df
            temp_list = pd.concat([class_one_df.head(proportion_to_one_class)])
            class_one_df = class_one_df.iloc[proportion_to_one_class:]

            temp_list = pd.concat([temp_list, class_two_df.head(proportion_to_two_class)])
            class_two_df = class_two_df.iloc[proportion_to_two_class:]

            temp_list = pd.concat([temp_list, class_three_df.head(proportion_to_three_class)])
            class_three_df = class_three_df.iloc[proportion_to_three_class:]

            lista.append(temp_list)

    return lista


def test_algorithms(data):
    """
        Pega as fold passados e os passa por knn, árvore de decisão e Naive Bayes Gaussiano.

        :param data: lista contendo dataframes sendo cada dataframe correspondente a um fold
    """
    for i in range(len(data)):
        folds = data.copy()
        test_df = folds[i].copy()
        del folds[i]
        concatenated = pd.concat(folds)
        train_x = concatenated[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                          'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                          'od280/od315_of_diluted_wines', 'proline']]
        train_y = concatenated.pop('class')
        test_x = test_df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                          'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                          'od280/od315_of_diluted_wines', 'proline']]
        test_y = test_df.pop('class')
        train_x_normalized = preprocessing.normalize(train_x, norm='l2')
        test_x_normalized = preprocessing.normalize(test_x, norm='l2')

        # Lembrar de testar diferentes parâmetros no treinamento para explicar no relatório os melhores
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        model = KNeighborsClassifier(n_neighbors=7)
        model.fit(train_x_normalized, train_y)
        prediction = model.predict(test_x_normalized)
        print('The accuracy of the KNN is', metrics.accuracy_score(prediction, test_y))

        model = DecisionTreeClassifier()
        model.fit(train_x_normalized, train_y)
        prediction = model.predict(test_x_normalized)
        print('The accuracy of the Decision Tree is', metrics.accuracy_score(prediction,test_y))

        gnb = GaussianNB()
        y_pred = gnb.fit(train_x_normalized, train_y).predict(test_x_normalized)
        # print("Number of mislabeled points out of a total %d points : %d" %
        #       (test_x.shape[0], (test_y != y_pred).sum()))
        print('The accuracy of the Naive Bayes Gaussian is: %f' %
              (1 - (((test_y != y_pred).sum()) / (test_x_normalized.shape[0]))))
        print('_________________________________________________________________________')

# def confusion_matrix():


list_folds = k_fold(3)
test_algorithms(list_folds)
