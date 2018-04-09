from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from nltk import word_tokenize
import numpy as np
import pandas as pd
from utils import accuracy, precision, recall, f1
import _pickle as cPickle
from os import listdir, path

np.random.seed(1)


kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)


def tokenize(text):
    return word_tokenize(text)


def train(x_train, y_train, check_trained=True, ngram_range=(1,1), vector_type="count"):
    fname = 'rf_'+vector_type+"_vectorized_model_"+"_".join([str(i) for i in ngram_range])+".pkl"
    if check_trained :
        for file in listdir('./ml_models/'):
            if file == fname:
                print("Model present, returning trained model")
                with open('./ml_models/'+fname,'rb') as fid:
                    return cPickle.load(fid)

    cls = RandomForestClassifier()
    grid_cls = GridSearchCV(cls,
                            param_grid={
                                'criterion': ['gini', 'entropy'],
                                'n_estimators': [10, 50, 100]
                            },
                            cv=kfolds,
                            n_jobs=-1
                            )
    grid_cls.fit(x_train, y_train)

    with open(path.join('./ml_models/',fname), 'wb') as fid:
        cPickle.dump(grid_cls.best_estimator_, fid)

    return grid_cls.best_estimator_


def predict(model, x_train):
    return model.predict(x_train)


def vectorized_rf(x_train, y_train, x_test, y_test, x_elec, y_elec, x_food, y_food, checktrain=True, ngram_range=(1, 1), vector_type="count"):
    vectorizer = CountVectorizer(tokenizer=tokenize, ngram_range=ngram_range) if vector_type=="count" \
        else TfidfVectorizer(tokenizer=tokenize, ngram_range=ngram_range)
    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)
    model = train(vectorized_x_train, y_train, checktrain, ngram_range, vector_type=vector_type)
    pred_x_train = predict(model, vectorized_x_train)
    pred_x_test = predict(model, vectorized_x_test)
    pred_elec = predict(model, vectorizer.transform(x_elec))
    pred_food = predict(model, vectorizer.transform(x_food))

    precision_test = precision(y_test, pred_x_test)
    recall_test = recall(y_test, pred_x_test)
    f1_test = f1(y_test, pred_x_test)

    print("Accuracy training accuracy (" + vector_type, " vectorized joint data) =", accuracy(y_train, pred_x_train))
    print("Accuracy testing accuracy (" + vector_type, "vectorized joint data) =", accuracy(y_test, pred_x_test), "\n")

    print("Precision (" + vector_type + " vectorized test data) =", precision_test)
    print("Recall (" + vector_type + " test data) =", recall_test)
    print("F1 (" + vector_type + " test data) =", f1_test, "\n")

    print("Accuracy electronics data ("+vector_type, "vectorized) =", accuracy(y_elec, pred_elec))
    print("Accuracy food data (" + vector_type, "vectorized) =", accuracy(y_food, pred_food),"\n")

    # Set confusion matrix for elec data and then compute precision, recall and f1_score
    precision_elec = precision(y_elec, pred_elec)
    recall_elec = recall(y_elec, pred_elec)
    f1_elec = f1(y_elec, pred_elec)

    # Set confusion matrix for food data and then compute precision, recall and f1_score
    precision_food = precision(y_food, pred_food)
    recall_food = recall(y_food, pred_food)
    f1_food = f1(y_food, pred_food)

    print("Precision ("+vector_type+" vectorized electricity data) =", precision_elec)
    print("Recall ("+vector_type+" electricity data) =", recall_elec)
    print("F1 ("+vector_type+" electricity data) =", f1_elec, "\n")

    print("Precision ("+vector_type+" food data) =", precision_food)
    print("Recall ("+vector_type+" food data) =", recall_food)
    print("F1 ("+vector_type+" food data) =", f1_food,"\n\n")


def main():
    joint_data_train = pd.read_csv('./data/ml_joint_data_train.csv')
    joint_data_test = pd.read_csv('./data/ml_joint_data_test.csv')
    elec_data = pd.read_csv('./data/ml_elec_data.csv')
    food_data = pd.read_csv('./data/ml_food_data.csv')

    x_train = joint_data_train['noaspect_text'].values.astype('U')
    y_train = joint_data_train['class'].values.astype(np.float32)
    x_test = joint_data_test['noaspect_text'].values.astype('U')
    y_test = joint_data_test['class'].values.astype(np.float32)
    x_elec = elec_data['noaspect_text'].values.astype('U')
    y_elec = elec_data['class'].values.astype(np.float32)
    x_food = food_data['noaspect_text'].values.astype('U')
    y_food = food_data['class'].values.astype(np.float32)

    vectorized_rf(x_train, y_train, x_test, y_test, x_elec, y_elec, x_food, y_food, ngram_range=(1,1), vector_type="count")
    vectorized_rf(x_train, y_train, x_test, y_test, x_elec, y_elec, x_food, y_food, ngram_range=(1,1), vector_type="tfidf")


if __name__ == '__main__':
    main()