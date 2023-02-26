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
    temp_list = []

    # Se for última iteração pega tudo que sobrou, mesmo que fique fold maior
    if(i == k):
      temp_list.append(class_one_df)
      temp_list.append(class_two_df)
      temp_list.append(class_three_df)
      lista.append(temp_list)

    else:
        temp_list.append(class_one_df.head(proportion_to_one_class))
        class_one_df = class_one_df.iloc[proportion_to_one_class:]

        temp_list.append(class_two_df.head(proportion_to_two_class))
        class_two_df = class_two_df.iloc[proportion_to_two_class:]

        temp_list.append(class_three_df.head(proportion_to_three_class))
        class_three_df = class_three_df.iloc[proportion_to_three_class:]

        lista.append(temp_list)

k_fold(3)