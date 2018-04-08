from spacy.tokenizer import Tokenizer
from spacy import load
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

number_batch = './data/numberbatch-en.txt'

nlp = load("en")
tokenizer = Tokenizer(nlp.vocab)


# Remove Stopwords
def remove_stopwords(sentence) :
    return " ".join([str(token) for token in tokenizer(sentence.replace('[comma]', '').replace(".","").lower())
                     if not token.is_stop and not token.is_punct and not token.is_digit and token.is_alpha])


# split and get left side of the sentence
def split_left(text_splitpoint) :
    sentence, split_point = text_splitpoint
    return sentence.split(split_point)[0]


# split and get right side of the sentence
def split_right(text_splitpoint):
    sentence, split_point = text_splitpoint
    split = sentence.split(split_point)
    return split[1] if len(split)>1 else " "


class Sensitivity:
    __instance = None

    _tp = {}
    _fp = {}
    _tn = {}
    _fn = {}
    _flushed = True

    _labels = ()
    _precision = {}
    _recall = {}
    _f1 = {}

    y_train = np.empty(shape=(0, 0))

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(Sensitivity)
        return cls.__instance

    def set_matrix(self, actuals, prediction, labels=(-1, 0, 1)):
        self.flush()
        self.y_train = actuals.copy()
        self._labels = labels
        matrix = confusion_matrix(actuals, prediction, labels=labels)
        for i, label in enumerate(labels):
            self._tp[label] = matrix[i,i]
            self._fn[label] = np.sum(matrix[i,:]) - self._tp[label]
            self._fp[label] = np.sum(matrix[:,i]) - self._tp[label]
            self._tn[label] = np.sum(matrix) - self._tp[label] - self._fn[label] - self._fp[label]
        self._flushed = False

    def get_set_precision(self):
        if self._precision is not None and not self._precision == {}:
            return self._precision
        if self._flushed:
            raise RuntimeError("Set Confusion matrix first")

        for i, label in enumerate(self._labels):
            self._precision[label] = 0 if self._tp[label] == 0 else self._tp[label]/(self._tp[label] + self._fp[label])
        return self._precision

    def get_set_recall(self):
        if self._recall is not None and not self._recall == {}:
            return self._recall
        if self._flushed:
            raise RuntimeError("Set Confusion matrix first")

        for i, label in enumerate(self._labels):
            self._recall[label] = 0 if self._tp[label] == 0 else self._tp[label] / (self._tp[label] + self._fn[label])
        return self._recall

    def get_set_f1(self):
        if self._f1 is not None and not self._f1 == {}:
            return self._f1

        if self._flushed:
            raise RuntimeError("Set confusion matrix first")

        if self._precision is None or self._precision == {} :
            self.get_set_precision()

        if self._recall is None or self._recall == {} :
            self.get_set_recall()

        for i, label in enumerate(self._labels):
            numerator = self._precision[label]*self._recall[label]*2
            denominator = self._precision[label] + self._recall[label]
            self._f1[label] = 0 if numerator == 0 else numerator/denominator
        return self._f1

    def flush(self):
        self._tp = {}
        self._fp = {}
        self._tn = {}
        self._fn = {}

        self._labels = ()
        self._precision = {}
        self._recall = {}
        self._f1 = {}
        self.y_train = np.empty(shape=(0, 0))
        self._flushed = True

    def is_flushed(self):
        return self._flushed


# Accuracy
def accuracy(actuals, predictions, labels=(-1, 0, 1)) :
    return accuracy_score(y_true=actuals, y_pred=predictions)


# Precision
def precision(actuals, predictions, labels=(-1, 0, 1)) :
    sensitivity = Sensitivity()
    if sensitivity.is_flushed() or not np.array_equal(sensitivity.y_train, actuals):
        sensitivity.set_matrix(actuals, predictions, labels)
    return sensitivity.get_set_precision()


# Recall
def recall(actuals, predictions, labels=(-1, 0, 1)) :
    sensitivity = Sensitivity()
    if sensitivity.is_flushed() or not np.array_equal(sensitivity.y_train, actuals):
        sensitivity.set_matrix(actuals, predictions, labels)
    return sensitivity.get_set_recall()


# F1 score
def f1(actuals, predictions, labels=(-1, 0, 1)):
    sensitivity = Sensitivity()
    if sensitivity.is_flushed() or not np.array_equal(sensitivity.y_train, actuals):
        sensitivity.set_matrix(actuals, predictions, labels)
    return sensitivity.get_set_f1()
