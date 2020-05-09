This is a python 3.6 code to implement learning decision tree algorithm to classify a labeled data.
part_one is associated with discrete data of a restaurant.
part_one is associated with continuous data of diabetes.

this code will find a classification method through learning decision tree algorithm by
using information gain metric

INPUTS:
No inputs needed for this implementation. The address of data is already set in code.
OUTPUTS:
1. Decision Tree
2. Information gain & branches entropy at each node.
3. Population in leaves.
4. Accuracy of tree on test data. which is 20% of whole data


CONFIGURATIONS
TARGET = 'Outcome' #name of the column of output
###########################################################
PRIME = '_prime' #used for changing continuous data to discrete
###########################################################
Q = 'q' #used for changing continuous data to discrete
###########################################################
YES = 1 #how data set shown true values in outcome
NO = 0 #how data set has shown false value in outcome
###########################################################
N = 2 #number of classes between min & max of continuous data
###########################################################
DATASET_ADDRESS = "diabetes.csv" #address of data set
###########################################################
# headers of data set
headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']