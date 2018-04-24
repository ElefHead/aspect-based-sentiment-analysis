from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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


def train(x_train, y_train, check_trained=True, ngram_range=(1,1), vector_type="count", dataset="default"):
    fname = dataset + '_rf_' + vector_type + "_vectorized_model_" + "_".join([str(i) for i in ngram_range]) + ".pkl"
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

    print(grid_cls.best_params_)

    with open(path.join('./ml_models/',fname), 'wb') as fid:
        cPickle.dump(grid_cls.best_estimator_, fid)

    return grid_cls.best_estimator_


def predict(model, x_train):
    return model.predict(x_train)


def vectorized_rf(x_train, y_train, x_test, y_test, checktrain=True, ngram_range=(1, 1), vector_type="count", dataset="default"):
    vectorizer = CountVectorizer(tokenizer=tokenize, ngram_range=ngram_range) if vector_type=="count" \
        else TfidfVectorizer(tokenizer=tokenize, ngram_range=ngram_range)
    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)
    model = train(vectorized_x_train, y_train, checktrain, ngram_range, vector_type=vector_type, dataset=dataset)
    pred_x_train = predict(model, vectorized_x_train)
    pred_x_test = predict(model, vectorized_x_test)

    precision_test = precision(y_test, pred_x_test)
    recall_test = recall(y_test, pred_x_test)
    f1_test = f1(y_test, pred_x_test)

    print("Accuracy training accuracy (" + dataset, vector_type, " vectorized joint data) =", accuracy(y_train, pred_x_train))
    print("Accuracy testing accuracy (" + dataset, vector_type, "vectorized joint data) =", accuracy(y_test, pred_x_test), "\n")

    print("Precision (" + dataset, vector_type + " vectorized test data) =", precision_test)
    print("Recall (" + dataset, vector_type + " test data) =", recall_test)
    print("F1 (" + dataset, vector_type + " test data) =", f1_test, "\n")


def main():

    elec_data = pd.read_csv('./data/ml_elec_data.csv')
    food_data = pd.read_csv('./data/ml_food_data.csv')

    x_electrain, x_electest, y_electrain, y_electest = train_test_split(elec_data['noaspect_text'], elec_data['class'], shuffle=True, stratify=elec_data['class'])

    vectorized_rf(x_electrain.values.astype('U'), y_electrain.values.astype(np.float32), x_electest.values.astype('U'), y_electest.values.astype(np.float32), ngram_range=(1,1), vector_type="count", dataset="elec")
    vectorized_rf(x_electrain.values.astype('U'), y_electrain.values.astype(np.float32), x_electest.values.astype('U'), y_electest.values.astype(np.float32), ngram_range=(1,1), vector_type="tfidf", dataset="elec")

    x_foodtrain, x_foodtest, y_foodtrain, y_foodtest = train_test_split(food_data['noaspect_text'], food_data['class'], shuffle=True, stratify=food_data['class'])

    vectorized_rf(x_foodtrain.values.astype('U'), y_foodtrain.values.astype(np.float32), x_foodtest.values.astype('U'),
                  y_foodtest.values.astype(np.float32), ngram_range=(1, 1), vector_type="count", dataset="food")
    vectorized_rf(x_foodtrain.values.astype('U'), y_foodtrain.values.astype(np.float32), x_foodtest.values.astype('U'),
                  y_foodtest.values.astype(np.float32), ngram_range=(1, 1), vector_type="tfidf", dataset="food")


if __name__ == '__main__':
    main()