"""
    Part 1 of second implementation homework.
    I used following resources to implement this part
    https://www.youtube.com/watch?v=LDRbO9a6XPU
"""
import pandas as pd
import numpy as np

# reading data from csv file
training_data = pd.read_csv("restaurant.csv")
# header = training_data.columns
headers = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type',
           'Est', 'WillWait']
TARGET = 'WillWait'


class Question:
    """ Question is called to partition the data set.
    This class keeps feature name and possible values for that feature in a given data set.
    """

    def __init__(self, column, df):
        self.column = column
        self.column_possible_values = df[column].value_counts().index.tolist()
        self.num_possible_values = len(df[column].value_counts().index.tolist())

    def partition(self, df):
        """Partitions a data set.
        For each row in the data set, check if it matches one of the possible values.
        """
        output = list()
        for possible_value in self.column_possible_values:
            temp = df[df[self.column] == possible_value]
            output.append(temp)
        return output

    def __str__(self):
        return "??" + str(self.column) + "??"


#######
# Demo:
# Let's write a question for a Est attribute \
# q = Question('Est', training_data)
# partitions = q.partition(training_data)
# for partition in partitions:
#     print(partition.head())
#     print("******\n\n")
#######

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
    for sub_df in list_of_partitions:
        n_k_and_pk = sub_df.shape[0]
        pk = sub_df[sub_df[TARGET] == "Yes"].shape[0]
        nk = n_k_and_pk - pk
        r += remainder_util(total_elements, nk, pk)
    return r


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
    return B(p / (n + p)) - reminder(p + n, list_of_partitions)


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
    p = df[df[TARGET] == "Yes"].shape[0]
    n = n_and_p - p

    for feature in headers[:-1]:
        question = Question(feature, df)
        info_gained = gain(n, p, question.partition(df))
        if info_gained > best_gain:
            best_gain, best_question = info_gained, question
    return best_gain, best_question


#######
# Demo:
# print(find_best_split(training_data))
#######


def leaf_count(df):
    """
    Counts the number of No & Yes samples in a given data set.
    """
    unique_values_count = df[TARGET].value_counts().array
    unique_values_name = df[TARGET].value_counts().index.tolist()
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
    partitions_nodes = [] # to keep reference to children nodes
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
    print(spacing1 + str(node.question))
    for i, possible_value in enumerate(node.question.column_possible_values):
        # Call this function recursively on the branches
        print(spacing1 + '--> :' + str(possible_value))
        print_tree(node.branches[i], spacing1 + spacing2)


#######
# Demo:
my_tree = build_tree(training_data)
print(isinstance(my_tree, MiddleNode))
print_tree(my_tree)
#######
