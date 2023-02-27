import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

        # Lembrar de testar diferentes parâmetros no treinamento para explicar no relatório os melhores
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

        print('_________________________________________________________________________')

    plot_acc_dict = {
        'knn': acc_knn_list,
        'decision tree': acc_decision_tree_list,
        'naive bayes': acc_naive_bayes_list
    }

    fig, ax = plt.subplots()
    ax.boxplot(plot_acc_dict.values())
    ax.set_xticklabels(plot_acc_dict.keys())
    plt.title('Acurácia')
    plt.show()

    plot_precision_dict = {
        'knn': precision_knn_list,
        'decision tree': precision_decision_tree_list,
        'naive bayes': precision_naive_bayes_list
    }

    fig, ax = plt.subplots()
    ax.boxplot(plot_precision_dict.values())
    ax.set_xticklabels(plot_precision_dict.keys())
    plt.title('Precisão')
    plt.show()

    plot_rev_dict = {
        'knn': rev_knn_list,
        'decision tree': rev_decision_tree_list,
        'naive bayes': rev_naive_bayes_list
    }

    fig, ax = plt.subplots()
    ax.boxplot(plot_rev_dict.values())
    ax.set_xticklabels(plot_rev_dict.keys())
    plt.title('Revogação')
    plt.show()

    plot_f1_dict = {
        'knn': f1_knn_list,
        'decision tree': f1_decision_tree_list,
        'naive bayes': f1_naive_bayes_list
    }

    fig, ax = plt.subplots()
    ax.boxplot(plot_f1_dict.values())
    ax.set_xticklabels(plot_f1_dict.keys())
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
    model = KNeighborsClassifier(n_neighbors=7)
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

    # print('Acc = ', acc_mean)
    # print('Precisão = ', precision_mean)
    # print('Revogação = ', rev_mean)
    # print('F1 = ', f1_mean)
    # print('                            ')

    return acc_mean, precision_mean, rev_mean, f1_mean


list_folds = k_fold(3)
test_algorithms(list_folds)
