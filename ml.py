import pandas as pd  # utilizado para tratar os dados obtidos a partir do csv
import numpy as np  # cálculo de média e desvio padrão
import matplotlib.pyplot as plt  # plot do histograma e boxplot dos desempenhos
import seaborn as sns  # plot do gráfico de violino e mapa de calor
from sklearn.model_selection import train_test_split  # fazer o holdout
import sklearn.preprocessing as preprocessing  # normalização dos dados
from sklearn.neighbors import KNeighborsClassifier  # método knn
from sklearn.tree import DecisionTreeClassifier  # método de árvore de decisão
from sklearn.naive_bayes import GaussianNB  # Naive Bayes Gaussiano
from sklearn.tree import plot_tree  # plot da árvore

df = pd.read_csv('wine.csv')
X = df.drop(['class'], axis=1)
y = df['class'].values
train_atributes, test_atributes, train_classes, test_classes = train_test_split(X, y, test_size=0.20,
                                                                                stratify=y, random_state=42)
test_dataset = train_atributes.copy()
test_dataset['class'] = train_classes

# pairplot da distribuição de instâncias por atributo
# sns.pairplot(df, hue="class")
# plt.show()

# gráfico histograma
# X.hist()
# fig = plt.gcf()
# plt.show()

# gráfico violino
# df.rename(columns={'class': 'tipo de uva'}, inplace=True)
# plt.subplot(3, 5, 1)
# sns.violinplot(x='tipo de uva', y='alcohol', data=df)
# plt.subplot(3, 5, 2)
# sns.violinplot(x='tipo de uva', y='malic_acid', data=df)
# plt.subplot(3, 5, 3)
# sns.violinplot(x='tipo de uva', y='ash', data=df)
# plt.subplot(3, 5, 4)
# sns.violinplot(x='tipo de uva', y='alcalinity_of_ash', data=df)
# plt.subplot(3, 5, 5)
# sns.violinplot(x='tipo de uva', y='magnesium', data=df)
# plt.subplot(3, 5, 6)
# sns.violinplot(x='tipo de uva', y='total_phenols', data=df)
# plt.subplot(3, 5, 7)
# sns.violinplot(x='tipo de uva', y='flavanoids', data=df)
# plt.subplot(3, 5, 8)
# sns.violinplot(x='tipo de uva', y='nonflavanoid_phenols', data=df)
# plt.subplot(3, 5, 9)
# sns.violinplot(x='tipo de uva', y='proanthocyanins', data=df)
# plt.subplot(3, 5, 10)
# sns.violinplot(x='tipo de uva', y='color_intensity', data=df)
# plt.subplot(3, 5, 11)
# sns.violinplot(x='tipo de uva', y='hue', data=df)
# plt.subplot(3, 5, 12)
# sns.violinplot(x='tipo de uva', y='od280/od315_of_diluted_wines', data=df)
# plt.subplot(3, 5, 13)
# sns.violinplot(x='tipo de uva', y='proline', data=df)
# plt.show()

# heatmap
# sns.heatmap(X.corr(), annot=True)
# plt.show()


def k_fold(k, dataframe):
    """
        Gera k folds de um dataframe
        :param k: número de folds
        :param dataframe: dataframe a ser utilizado para gerar os folds
        :return: lista: lista contendo dataframes sendo cada dataframe correspondente a um fold
    """
    class_one_df = dataframe.loc[dataframe['class'] == 1]
    class_two_df = dataframe.loc[dataframe['class'] == 2]
    class_three_df = dataframe.loc[dataframe['class'] == 3]
    fold_length = round(len(dataframe)/k)
    proportion_to_one_class = round(len(class_one_df)/len(dataframe) * fold_length)
    proportion_to_two_class = round(len(class_two_df)/len(dataframe) * fold_length)
    proportion_to_three_class = round(len(class_three_df)/len(dataframe) * fold_length)

    lista = []

    for i in range(k):
        # Se for última iteração pega tudo que sobrou, mesmo que fique fold de tamanho diferente
        if i == k-1:
            temp_list = pd.concat([class_one_df])
            temp_list = pd.concat([temp_list, class_two_df])
            temp_list = pd.concat([temp_list, class_three_df])
            lista.append(temp_list)
        else:
            temp_list = class_one_df.sample(proportion_to_one_class)
            class_one_df = class_one_df.drop(temp_list.index.values.tolist())

            class_two_sample = class_two_df.sample(proportion_to_two_class)
            temp_list = pd.concat([temp_list, class_two_sample])
            class_two_df = class_two_df.drop(class_two_sample.index.values.tolist())

            class_three_sample = class_three_df.sample(proportion_to_three_class)
            temp_list = pd.concat([temp_list, class_three_sample])
            class_three_df = class_three_df.drop(class_three_sample.index.values.tolist())

            lista.append(temp_list)
    return lista


def cross_validation(data):
    """
        Pega as fold passados e os passa por knn, árvore de decisão e Naive Bayes Gaussiano.

        :param data: lista contendo dataframes sendo cada dataframe correspondente a um fold
    """
    acc_knn_list = []
    acc_decision_tree_list = []
    acc_naive_bayes_list = []

    precision_knn_list = []
    precision_decision_tree_list = []
    precision_naive_bayes_list = []

    rev_knn_list = []
    rev_decision_tree_list = []
    rev_naive_bayes_list = []

    f1_knn_list = []
    f1_decision_tree_list = []
    f1_naive_bayes_list = []

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

        print('Teste com o fold', i+1)
        print('KNN')
        prediction = knn(train_x_normalized, train_y, test_x_normalized)
        matrix = confusion_matrix(prediction, test_y.tolist())
        evaluation_metrics = calculate_evaluation_metrics(matrix)
        mean_list = calc_mean(evaluation_metrics)

        acc_knn_list.append(mean_list[0])
        precision_knn_list.append(mean_list[1])
        rev_knn_list.append(mean_list[2])
        f1_knn_list.append(mean_list[3])

        print('Decision Tree')
        prediction = decision_tree(train_x_normalized, train_y, test_x_normalized)
        matrix = confusion_matrix(prediction, test_y.tolist())
        evaluation_metrics = calculate_evaluation_metrics(matrix)
        mean_list = calc_mean(evaluation_metrics)

        acc_decision_tree_list.append(mean_list[0])
        precision_decision_tree_list.append(mean_list[1])
        rev_decision_tree_list.append(mean_list[2])
        f1_decision_tree_list.append(mean_list[3])

        print('Naive Bayes')
        prediction = naive_bayes_gaussian(train_x_normalized, train_y, test_x_normalized)
        matrix = confusion_matrix(prediction, test_y.tolist())
        evaluation_metrics = calculate_evaluation_metrics(matrix)
        mean_list = calc_mean(evaluation_metrics)

        acc_naive_bayes_list.append(mean_list[0])
        precision_naive_bayes_list.append(mean_list[1])
        rev_naive_bayes_list.append(mean_list[2])
        f1_naive_bayes_list.append(mean_list[3])

        print('______________________________________________________________________________')

    print('Média e desvio padrão dos algoritmos pelo cross validation:')
    print(' ')
    print('acc_knn_mean and std_dev:', np.mean(acc_knn_list), np.std(acc_knn_list))
    print('acc_decision_tree_list and std_dev:', np.mean(acc_decision_tree_list), np.std(acc_decision_tree_list))
    print('acc_naive_bayes_list and std_dev:', np.mean(acc_naive_bayes_list), np.std(acc_naive_bayes_list))
    print('   ')
    print('precision_knn_list and std_dev:', np.mean(precision_knn_list), np.std(precision_knn_list))
    print('precision_decision_tree_list and std_dev:', np.mean(precision_decision_tree_list),
          np.std(precision_decision_tree_list))
    print('precision_naive_bayes_list and std_dev:', np.mean(precision_naive_bayes_list),
          np.std(precision_naive_bayes_list))
    print('   ')
    print('rev_knn_list and std_dev:', np.mean(rev_knn_list), np.std(rev_knn_list))
    print('rev_decision_tree_list and std_dev:', np.mean(rev_decision_tree_list), np.std(rev_decision_tree_list))
    print('rev_naive_bayes_list and std_dev:', np.mean(rev_naive_bayes_list), np.std(rev_naive_bayes_list))
    print('   ')
    print('f1_knn_list and std_dev:', np.mean(f1_knn_list), np.std(f1_knn_list))
    print('f1_decision_tree_list and std_dev:', np.mean(f1_decision_tree_list), np.std(f1_decision_tree_list))
    print('f1_naive_bayes_list and std_dev:', np.mean(f1_naive_bayes_list), np.std(f1_naive_bayes_list))
    print('______________________________________________________________________________')

    plot_acc_dict = {
        'knn': acc_knn_list,
        'decision tree': acc_decision_tree_list,
        'naive bayes': acc_naive_bayes_list
    }

    fig1, ax1 = plt.subplots()
    ax1.boxplot(plot_acc_dict.values())
    ax1.set_xticklabels(plot_acc_dict.keys())
    plt.title('Acurácia')

    plot_precision_dict = {
        'knn': precision_knn_list,
        'decision tree': precision_decision_tree_list,
        'naive bayes': precision_naive_bayes_list
    }

    fig2, ax2 = plt.subplots()
    ax2.boxplot(plot_precision_dict.values())
    ax2.set_xticklabels(plot_precision_dict.keys())
    plt.title('Precisão')

    plot_rev_dict = {
        'knn': rev_knn_list,
        'decision tree': rev_decision_tree_list,
        'naive bayes': rev_naive_bayes_list
    }

    fig3, ax3 = plt.subplots()
    ax3.boxplot(plot_rev_dict.values())
    ax3.set_xticklabels(plot_rev_dict.keys())
    plt.title('Revocação')

    plot_f1_dict = {
        'knn': f1_knn_list,
        'decision tree': f1_decision_tree_list,
        'naive bayes': f1_naive_bayes_list
    }

    fig4, ax4 = plt.subplots()
    ax4.boxplot(plot_f1_dict.values())
    ax4.set_xticklabels(plot_f1_dict.keys())
    plt.title('F1')
    plt.show()


def knn(train_x, train_y, test_x):
    """
        Utiliza o algoritmo knn para fazer a previsão de classe

        :param train_x: valores dos atributos de treino
        :param train_y: valor da classe de treino
        :param test_x: valor dos atributos de teste
        :return: prediction: predição de classe calculada para os valores de teste
    """
    model = KNeighborsClassifier(n_neighbors=7, p=1)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def decision_tree(train_x, train_y, test_x):
    """
        Utiliza o algoritmo decision tree para fazer a previsão de classe

        :param train_x: valores dos atributos de treino
        :param train_y: valor da classe de treino
        :param test_x: valor dos atributos de teste
        :return: prediction: predição de classe calculada para os valores de teste
    """
    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    # plot_tree(model)
    # plt.show()
    return model.predict(test_x)


def naive_bayes_gaussian(train_x, train_y, test_x):
    """
        Utiliza o algoritmo naive bayes gaussian para fazer a previsão de classe

        :param train_x: valores dos atributos de treino
        :param train_y: valor da classe de treino
        :param test_x: valor dos atributos de teste
        :return: prediction: predição de classe calculada para os valores de teste
    """
    gnb = GaussianNB()
    return gnb.fit(train_x, train_y).predict(test_x)


def confusion_matrix(predicted_class, correct_class):
    """
        Gera uma matriz de confusão para os valores de classe passados

        :param predicted_class: lista contendo os valores preditos para a classe de interesse
        :param correct_class: lista contendo os valores corretos para a classe de interesse
        :return: matrix: matriz de confusão calculada
    """
    matrix = [[0 for i in range(3)] for j in range(3)]
    for i in range(len(predicted_class)):
        for j in range(3):
            if predicted_class[i] == j+1 and correct_class[i] == 1:
                matrix[0][j] += 1
            if predicted_class[i] == j+1 and correct_class[i] == 2:
                matrix[1][j] += 1
            if predicted_class[i] == j+1 and correct_class[i] == 3:
                matrix[2][j] += 1

    return matrix


def calculate_evaluation_metrics(matrix):
    """
        Calcula as métrica de avaliação

        :param matrix: matriz de confusão
        :return: lista: cada valor na lista contém um objeto com as métricas para cada classe
    """
    class_1_metrics = {
        'vp': matrix[0][0],
        'vn': matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix[2][2],
        'fp': matrix[1][0] + matrix[2][0],
        'fn': matrix[0][1] + matrix[0][2]
    }
    class_2_metrics = {
        'vp': matrix[1][1],
        'vn': matrix[0][0] + matrix[0][2] + matrix[2][0] + matrix[2][2],
        'fp': matrix[0][1] + matrix[2][1],
        'fn': matrix[1][0] + matrix[1][2]
    }
    class_3_metrics = {
        'vp': matrix[2][2],
        'vn': matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1],
        'fp': matrix[0][2] + matrix[1][2],
        'fn': matrix[2][0] + matrix[2][1]
    }
    return class_1_metrics, class_2_metrics, class_3_metrics


def acc(class_metrics):
    """
        Calcula a acurácia

        :param class_metrics: métricas para o cálculo
        :return: value: valor calculado da acurácia
    """
    total = class_metrics['vp'] + class_metrics['vn'] + class_metrics['fp'] + class_metrics['fn']
    return (class_metrics['vp'] + class_metrics['vn']) / total


def precision(class_metrics):
    """
        Calcula a precisão

        :param class_metrics: métricas para o cálculo
        :return: value: valor calculado da precisão
    """
    return class_metrics['vp'] / (class_metrics['vp'] + class_metrics['fp'])


def rev(class_metrics):
    """
        Calcula a revogação

        :param class_metrics: métricas para o cálculo
        :return: value: valor calculado da revogação
    """
    return class_metrics['vp'] / (class_metrics['vp'] + class_metrics['fn'])


def f1(class_metrics):
    """
        Calcula a F1

        :param class_metrics: métricas para o cálculo
        :return: value: valor calculado da F1
    """
    return 2 * precision(class_metrics) * rev(class_metrics) / (precision(class_metrics) + rev(class_metrics))


def calc_mean(evaluation_metrics):
    """
        Calcula a média macro das métricar

        :param evaluation_metrics: métricas para o cálculo
        :return: lista: acc_value, precision_value, rev_value, f1_value
    """
    acc_value = 0
    precision_value = 0
    rev_value = 0
    f1_value = 0

    for j in range(len(evaluation_metrics)):
        acc_value += acc(evaluation_metrics[j])
        precision_value += precision(evaluation_metrics[j])
        rev_value += rev(evaluation_metrics[j])
        f1_value += f1(evaluation_metrics[j])

    acc_mean = acc_value / len(evaluation_metrics)
    precision_mean = precision_value / len(evaluation_metrics)
    rev_mean = rev_value / len(evaluation_metrics)
    f1_mean = f1_value / len(evaluation_metrics)

    print('Acc = ', acc_mean)
    print('Precisão = ', precision_mean)
    print('Revogação = ', rev_mean)
    print('F1 = ', f1_mean)
    print('                            ')

    return acc_mean, precision_mean, rev_mean, f1_mean


def final_test(train_x, test_x, train_y, test_y):
    train_x_normalized = preprocessing.normalize(train_x, norm='l2')
    test_x_normalized = preprocessing.normalize(test_x, norm='l2')

    acc_knn_list = []
    acc_decision_tree_list = []
    acc_naive_bayes_list = []

    precision_knn_list = []
    precision_decision_tree_list = []
    precision_naive_bayes_list = []

    rev_knn_list = []
    rev_decision_tree_list = []
    rev_naive_bayes_list = []

    f1_knn_list = []
    f1_decision_tree_list = []
    f1_naive_bayes_list = []

    print('Teste final com o holdout de 80/20')
    print('KNN')
    prediction = knn(train_x_normalized, train_y, test_x_normalized)
    matrix = confusion_matrix(prediction, test_y.tolist())
    evaluation_metrics = calculate_evaluation_metrics(matrix)
    mean_list = calc_mean(evaluation_metrics)

    acc_knn_list.append(mean_list[0])
    precision_knn_list.append(mean_list[1])
    rev_knn_list.append(mean_list[2])
    f1_knn_list.append(mean_list[3])

    print('Decision Tree')
    prediction = decision_tree(train_x_normalized, train_y, test_x_normalized)
    matrix = confusion_matrix(prediction, test_y.tolist())
    evaluation_metrics = calculate_evaluation_metrics(matrix)
    mean_list = calc_mean(evaluation_metrics)

    acc_decision_tree_list.append(mean_list[0])
    precision_decision_tree_list.append(mean_list[1])
    rev_decision_tree_list.append(mean_list[2])
    f1_decision_tree_list.append(mean_list[3])

    print('Naive Bayes')
    prediction = naive_bayes_gaussian(train_x_normalized, train_y, test_x_normalized)
    matrix = confusion_matrix(prediction, test_y.tolist())
    evaluation_metrics = calculate_evaluation_metrics(matrix)
    mean_list = calc_mean(evaluation_metrics)

    acc_naive_bayes_list.append(mean_list[0])
    precision_naive_bayes_list.append(mean_list[1])
    rev_naive_bayes_list.append(mean_list[2])
    f1_naive_bayes_list.append(mean_list[3])


list_folds = k_fold(5, test_dataset)
cross_validation(list_folds)

# Utiliza o holdout inicial de 80%/20% para fazer teste final
final_test(train_atributes, test_atributes, train_classes, test_classes)
