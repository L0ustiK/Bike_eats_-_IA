from ia_categorie import *

FILE_NAME = "data_set/train.json"


if __name__ == '__main__':
    logistic_regression_classifier(FILE_NAME)
    decision_tree_classifier(FILE_NAME)
    naive_bayes_models(FILE_NAME)