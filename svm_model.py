from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.pipeline import make_pipeline, Pipeline
from nltk import word_tokenize
from spacy import load
import numpy as np
import pandas as pd

np.random.seed(1)

nlp = load("en")

kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)


def tokenize(text):
    return word_tokenize(text)


def tfidfSVM(data_train, y_train):
    vectorizer = CountVectorizer(ngram_range=(1,1),
                                 analyzer='word',
                                 tokenizer=tokenize)

    pipeline_svm = make_pipeline(vectorizer,
                                 SVC(probability=True, kernel="linear", class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'svc__C': [0.01, 0.1, 1],
                                        'svc__kernel': ['linear', 'rbf'],
                                        },
                            cv=kfolds,
                            verbose=1,
                            n_jobs=-1
                            )
    grid_svm.fit(data_train, y_train)
    print(grid_svm.score(data_train, y_train))


def main():
    data = pd.read_csv("./data/joint_data.csv")
    X = data['filtered_text'].values
    Y = data['class']
    print(X[:5])
    data_train, y_train, data_test, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    print(data_train[:5])
    # tfidfSVM(data_train, y_train)


if __name__ == '__main__':
    main()