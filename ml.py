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
            temp_list = pd.concat([class_one_df.head(proportion_to_one_class)])
            class_one_df = class_one_df.iloc[proportion_to_one_class:]

            temp_list = pd.concat([temp_list, class_two_df.head(proportion_to_two_class)])
            class_two_df = class_two_df.iloc[proportion_to_two_class:]

            temp_list = pd.concat([temp_list, class_three_df.head(proportion_to_three_class)])
            class_three_df = class_three_df.iloc[proportion_to_three_class:]

            lista.append(temp_list)

    return lista


def test_algorithms(data):

    for i in range(len(data)):

        folds = data.copy()
        del folds[i]

        concatenated = pd.concat(folds)
        # remover a classe do concatenated para um Y, e transformar X e Y para lista para passar pro knn
        # usar products_list = df.values.tolist()
        print('concatenated = ', concatenated)
        print('_______________________')
        print('i = ', i)


# list_folds = k_fold(3)
# print(list_folds)
test_algorithms(k_fold(3))
