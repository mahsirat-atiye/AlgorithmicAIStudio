"""
    Part 2 of second implementation homework.
    I used following resources to implement this part
    https://www.youtube.com/watch?v=LDRbO9a6XPU
"""
import pandas as pd
# reading data from csv file
import numpy as np
# used for math parts
from sklearn.model_selection import train_test_split
# used this for split train and test data
from sklearn.metrics import confusion_matrix
# used this for evaluating tree
import matplotlib.pyplot as plt


# used to plot data

class Config:
    TARGET = 'Outcome'
    PRIME = '_prime'
    Q = 'q'
    YES = 1
    NO = 0
    N = 2
    DATASET_ADDRESS = "diabetes.csv"
    headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


whole_data = pd.read_csv(Config.DATASET_ADDRESS)
y = whole_data.copy(deep=True)[Config.TARGET]


def change_data_set(df, n):
    for column in df.columns:
        if column == Config.TARGET:
            continue

        def f(x):
            min_value = df[column].min()
            max_value = df[column].max()
            step = (max_value - min_value) / n
            a = min_value
            b = a + step
            if x < a:
                return str(Config.Q) + "0"
            for i in range(1, n):
                if b > x >= a:
                    return str(Config.Q) + str(i)
                a = b
                b = a + step

            if x >= a:
                return str(Config.Q) + str(n)

        df[column + str(Config.PRIME)] = df[column].apply(lambda x: f(x))
    return df


#######
# Demo:
# whole_data = change_data_set(whole_data, N)
for i in range(len(Config.headers)):
    if Config.headers[i] == Config.TARGET:
        continue
    Config.headers[i] = str(Config.headers[i]) + str(Config.PRIME)


# print(whole_data.head())
# print(headers)
#######


class Question:
    """ Question is called to partition the data set.
    This class keeps feature name and possible values for that feature in a given data set.
    """

    def __init__(self, column, df):
        self.column = column
        self.column_possible_values = df[column].value_counts().index.tolist()
        self.num_possible_values = len(df[column].value_counts().index.tolist())
        self.gain_information = 0
        self.branches_entropy = []

    def partition(self, df):
        """Partitions a data set.
        For each row in the data set, check if it matches one of the possible values.
        """
        output = list()
        for possible_value in self.column_possible_values:
            temp = df[df[self.column] == possible_value]
            output.append(temp)
        return output

    def match_index(self, row):
        """
        Matches the given row to index of one of the branches of the node.
        """
        # df_row = pd.DataFrame(row)
        # df_row = change_data_set(df_row, N)
        df_row = row
        row_answer_value = df_row[self.column]
        return self.column_possible_values.index(row_answer_value)

    def print(self, spacing):
        return "??" + str(self.column) + "??\n" + str(spacing) + " GI: " + str(
            self.gain_information) + "\n " + str(spacing) + "Branches entropy: " + str(self.branches_entropy)

    def __str__(self):
        return "??" + str(self.column) + "??"

def B(q):
    """
    Calculates entropy of a given probability.
    """
    if q == 1 or q == 0:
        return 0
    return round(-1 * (q * np.log2(q) + (1 - q) * np.log2(1 - q)), 10)


#######
# Demo:
# print(B(0.0))
#######

def remainder_util(total_elements, n_k, p_k):
    """
    Calculates the sub_remainder part of a specific partition
    """
    return round((p_k + n_k) / total_elements * B(p_k / (p_k + n_k)), 10)


#######
# Demo:
# print(remainder_util(12, 4, 2))
#######
def reminder(total_elements, list_of_partitions):
    """
    Calculates reminder of an attribute.
    @:param total_elements is sum of Yes & No samples before partitioning
    """
    r = 0
    partitions_entropy = []
    for sub_df in list_of_partitions:
        n_k_and_pk = sub_df.shape[0]
        pk = sub_df[sub_df[Config.TARGET] == Config.YES].shape[0]
        nk = n_k_and_pk - pk
        t = remainder_util(total_elements, nk, pk)
        r += t
        partitions_entropy.append(t)
    return r, partitions_entropy


#######
# Demo:
# print(reminder(12, Question('Pat', training_data).partition(training_data)))
#######
def gain(n, p, list_of_partitions):
    """
    Calculates Information Gain of a specific attribute.
    @:param n refers to num of No samples before partitioning
    @:param p refers to num of Yes samples before partitioning
    @:param list_of_partitions refers to partitions resulted by the attribute.
    """
    r, partitions_entropy = reminder(p + n, list_of_partitions)
    return (B(p / (n + p)) - r), partitions_entropy


#######
# Demo:
# print(gain(6, 6, Question('Type', training_data).partition(training_data)))
#######

def find_best_split(df):
    """
    Find the best question to ask by iteration over all features
    and calculating the information gain and selecting most informative question."""
    best_gain = 0
    best_question = None
    n_and_p = df.shape[0]
    p = df[df[Config.TARGET] == Config.YES].shape[0]
    n = n_and_p - p

    for feature in Config.headers[:-1]:
        question = Question(feature, df)
        question.gain_information, question.branches_entropy = gain(n, p, question.partition(df))
        if question.gain_information > best_gain:
            best_gain, best_question = question.gain_information, question
    return best_gain, best_question


#######
# Demo:
# print(find_best_split(training_data))
#######


def leaf_count(df):
    """
    Counts the number of No & Yes samples in a given data set.
    """
    unique_values_count = df[Config.TARGET].value_counts().array
    unique_values_name = df[Config.TARGET].value_counts().index.tolist()
    counts = {}
    for i, name in enumerate(unique_values_name):
        counts[name] = unique_values_count[i]

    return counts


#######
# Demo:
# print(class_counts(training_data))
#######

class Leaf:
    """
    A Leaf node in tree. it classifies data.
    It keeps a dictionary of result like: {Yes: 4, No: 1}
    """

    def __init__(self, df):
        self.predictions = leaf_count(df)

    def predict(self):
        max_occur = 0
        max_key = None
        for key in self.predictions.keys():
            if self.predictions[key] > max_occur:
                max_occur = self.predictions[key]
                max_key = key
        return max_key


class MiddleNode:
    """A Non-Leaf node in tree. This node asks a question.
    It keeps a reference to the question, and its children.
    """

    def __init__(self, question, branches):
        self.question = question
        self.branches = branches


def build_tree(df):
    """
    Builds the tree using divide and conquer approach.
    Base case: When no further information is gained. So no more question is needed.
    Find best split and call the function recursively for partitions.
    """

    gain, question = find_best_split(df)

    # Base case: no further information gain
    if gain == 0:
        return Leaf(df)

    partitions = question.partition(df)
    partitions_nodes = []  # to keep reference to children nodes
    for partition in partitions:
        temp = build_tree(partition)
        partitions_nodes.append(temp)

    return MiddleNode(question, partitions_nodes)


#######
# Demo:
# my_tree = build_tree(training_data)
#######


def print_tree(node, spacing1="\t", spacing2="\t\t"):
    # Base case: leaf
    if isinstance(node, Leaf):
        print(spacing1, node.predictions)
        return
    print(spacing1 + (node.question.print(spacing1)))
    for i, possible_value in enumerate(node.question.column_possible_values):
        # Call this function recursively on the branches
        print(spacing1 + '--> :' + str(possible_value))
        print_tree(node.branches[i], spacing1 + spacing2)


#######
# Demo:
# my_tree = build_tree(whole_data)
# print(isinstance(my_tree, MiddleNode))
# print_tree(my_tree)
#######


def classify(row, node):
    """Puts the given row into tree."""
    # Base case:
    if isinstance(node, Leaf):
        return node.predict()
    branch_index = node.question.match_index(row)
    return classify(row, node.branches[branch_index])


#######
# Demo:
# The tree predicts the second row of  training data is No.
# print(classify(whole_data.iloc[1], my_tree))
#######

def predict_test_set(df, node):
    predicted_y = []
    for row_index in range(df.shape[0]):
        y = classify(df.iloc[i], node)
        predicted_y.append(y)
    return predicted_y


def evaluate_model(df, node):
    y_true = df[Config.TARGET].tolist()
    y_predicted = predict_test_set(df, node)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_predicted).ravel()
    return tn, fp, fn, tp


def acc(tn, fp, fn, tp):
    return (tn + tp) / (tn + tp + fn + fp)


def balanced_acc(tn, fp, fn, tp):
    return (tp / (fp + tp) + tn / (tn + fn)) / 2


def plot(n, accs):
    # ax = plt.subplot(111)
    plt.plot(n, accs, 'ro', linewidth=1.0, label="accuracy")

    leg = plt.legend(loc='best', mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()


def check_for_best_n():
    n_values = [i for i in range(2, 10, 2)]
    accs = []
    for value in n_values:
        whole_data = pd.read_csv(Config.DATASET_ADDRESS)
        whole_data = change_data_set(whole_data, value)
        training_data, test_data, y_train, y_test = train_test_split(whole_data, y, test_size=0.2)
        my_tree = build_tree(training_data)
        # print_tree(my_tree)
        tn, fp, fn, tp = evaluate_model(test_data, my_tree)
        accuracy = acc(tn, fp, fn, tp)
        accs.append(accuracy)
        print("accuracy: ", accuracy)
    plot(n_values, accs)


# check_for_best_n()

if __name__ == '__main__':
    whole_data = pd.read_csv(Config.DATASET_ADDRESS)
    whole_data = change_data_set(whole_data, Config.N)
    training_data, test_data, y_train, y_test = train_test_split(whole_data, y, test_size=0.2)
    my_tree = build_tree(training_data)
    print_tree(my_tree)
    tn, fp, fn, tp = evaluate_model(test_data, my_tree)
    accuracy = acc(tn, fp, fn, tp)
    print("\n\n\n ACCURACY: ", accuracy)
    # tn, fp, fn, tp = evaluate_model(training_data, my_tree)
    # accuracy = acc(tn, fp, fn, tp)
    # print("\n\n\n ACCURACY: TRAIN ", accuracy)
