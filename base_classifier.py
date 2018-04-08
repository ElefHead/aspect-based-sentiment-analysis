import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *

np.random.seed(1)


class MaxClassClassifier:
    def __init__(self):
        self.max_class = None

    def train(self, y_train):
        '''
        find the most frequently occurring class
        y_train is only required, no need or x_train
        assuming y_train is a numpy array
        '''
        (labels, counts) = np.unique(y_train, return_counts=True)
        self.max_class = labels[np.argmax(counts)]

    def predict(self, data):
        '''
        Given max_class which is the label that occurs most frequently in the dataset,
        assign that label to all datapoints.
        '''
        return np.ones(data.shape[0])*self.max_class


def main():
    # load data
    joint_data = pd.read_csv('./data/joint_data.csv')
    elec_data = pd.read_csv('./data/elec_data_clean.csv')
    food_data = pd.read_csv('./data/food_data_clean.csv')

    # get required data and labels
    x_joint = joint_data['filtered_text'].values
    y_joint = np.array(joint_data['class'].values, dtype=np.float32)
    x_elec = elec_data['filtered_text'].values
    y_elec = np.array(elec_data['class'].values, dtype=np.float32)
    x_food = food_data['filtered_text'].values
    y_food = np.array(food_data['class'].values, dtype=np.float32)

    # define accuracy arrays
    accuracies_test = np.zeros([10, 1])
    accuracies_train = np.zeros([10, 1])

    # 10 tries at this
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x_joint, y_joint,
                                                            test_size=0.2, random_state=1, shuffle=True)
        cls = MaxClassClassifier()
        cls.train(y_train)

        pred_test = cls.predict(x_test)
        pred_train = cls.predict(x_train)

        accuracies_train[i] = accuracy(y_train, pred_train)
        accuracies_test[i] = accuracy(y_test, pred_test)

    pred_elec = cls.predict(x_elec)
    pred_food = cls.predict(x_food)
    accuracy_elec_data = accuracy(y_elec, pred_elec)
    accuracy_food_data = accuracy(y_food, pred_food)

    print("Average training accuracy (joint data) =", np.mean(accuracies_train))
    print("Average testing accuracy (joint data) =", np.mean(accuracies_test), "\n")
    print("Testing accuracy (electricity data) =", accuracy_elec_data)
    print("Testing accuracy (food data) =", accuracy_food_data, "\n")

    # Set confusion matrix for elec data and then compute precision, recall and f1_score
    precision_elec = precision(y_elec, pred_elec)
    recall_elec = recall(y_elec, pred_elec)
    f1_elec = f1(y_elec, pred_elec)

    # Set confusion matrix for food data and then compute precision, recall and f1_score
    precision_food = precision(y_food, pred_food)
    recall_food = recall(y_food, pred_food)
    f1_food = f1(y_food, pred_food)

    print("Precision (electricity data) =", precision_elec)
    print("Recall (electricity data) =", recall_elec)
    print("F1 (electricity data) =", f1_elec, "\n")

    print("Precision (food data) =", precision_food)
    print("Recall (food data) =", recall_food)
    print("F1 (food data) =", f1_food)


if __name__ == '__main__':
    main()
