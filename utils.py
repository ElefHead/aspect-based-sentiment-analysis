from spacy.tokenizer import Tokenizer
from spacy import load
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import re
from collections import Counter
import pandas as pd

nlp = load("en")
tokenizer = Tokenizer(nlp.vocab)


# Remove Stopwords
def remove_stopwords(sentence) :
    return " ".join([str(token) for token in tokenizer(sentence.replace('[comma]', '').replace(".","").lower())
                     if not token.is_stop and not token.is_punct and not token.is_digit and token.is_alpha])


def remove_aspect(text_aspect) :
    text, aspect = text_aspect
    pattern = '\s*'+aspect.replace('(', '\(').replace(')', '\)')+'\s*'
    return re.sub(pattern, ' ', text)


def replace_comma(text) :
    return text.replace('[comma]','')


# split and get left side of the sentence
def split_left(text_splitpoint) :
    sentence, split_point = text_splitpoint
    return sentence.split(split_point)[0]


# split and get right side of the sentence
def split_right(text_splitpoint):
    sentence, split_point = text_splitpoint
    split = sentence.split(split_point)
    return split[1] if len(split)>1 else " "


# replace aspect term with $T$
def point_aspect(text_splitpint):
    sentence, split_point = text_splitpint
    return sentence.replace(split_point, " $T$ ")


# From here on
# data reading and what not
def get_dataset_resources(data_file_name, sent_word2idx, target_word2idx, word_set, max_sent_len):
    ''' updates word2idx and word_set '''
    if len(sent_word2idx) == 0:
        sent_word2idx["<pad>"] = 0

    word_count = []
    sent_word_count = []
    target_count = []

    words = []
    sentence_words = []
    target_words = []

    # with open(data_file_name, 'r') as data_file:
    # lines = data_file.read().split('\n')
    df = pd.read_csv(data_file_name)
    for line_no in range(df.shape[0]):
        sentence = df['token_text'][line_no].replace("[comma]","")
        target = df['aspect_term'][line_no]

        sentence.replace("$T$", "")
        sentence = sentence.lower()
        target = target.lower()
        max_sent_len = max(max_sent_len, len(sentence.split()))
        sentence_words.extend(sentence.split())
        target_words.extend([target])
        words.extend(sentence.split() + target.split())

    sent_word_count.extend(Counter(sentence_words).most_common())
    target_count.extend(Counter(target_words).most_common())
    word_count.extend(Counter(words).most_common())

    for word, _ in sent_word_count:
        if word not in sent_word2idx:
            sent_word2idx[word] = len(sent_word2idx)

    for target, _ in target_count:
        if target not in target_word2idx:
            target_word2idx[target] = len(target_word2idx)

    for word, _ in word_count:
        if word not in word_set:
            word_set[word] = 1

    return max_sent_len


def get_embedding_matrix(embeddings, sent_word2idx,  target_word2idx, edim):
    ''' returns the word and target embedding matrix '''
    word_embed_matrix = np.zeros([len(sent_word2idx), edim], dtype = float)
    target_embed_matrix = np.zeros([len(target_word2idx), edim], dtype = float)

    for word in sent_word2idx:
        if word in embeddings:
            word_embed_matrix[sent_word2idx[word]] = embeddings[word]

    for target in target_word2idx:
        for word in target:
            if word in embeddings:
                target_embed_matrix[target_word2idx[target]] += embeddings[word]
        target_embed_matrix[target_word2idx[target]] /= max(1, len(target.split()))

    # print(type(word_embed_matrix))
    return word_embed_matrix, target_embed_matrix


def get_dataset(data_file_name, sent_word2idx, target_word2idx, embeddings):
    ''' returns the dataset'''
    sentence_list = []
    location_list = []
    target_list = []
    polarity_list = []


    # with open(data_file_name, 'r') as data_file:
        # lines = data_file.read().split('\n')
    df = pd.read_csv(data_file_name)
    for line_no in range(df.shape[0]):
        sentence = df['token_text'][line_no].lower().replace("[comma]","")
        # print(sentence)
        target = df['aspect_term'][line_no].lower()
        polarity = int(df['class'][line_no])
        polarity = 2 if polarity == -1 else polarity

        sent_words = re.split(r"\s+",sentence)
        target_words = re.split(r"\s+", target)
        try:
            target_location = sent_words.index("$t$")
        except:
            print(sentence)
            print("sentence does not contain target element tag")
            continue
            # exit()

        id_tokenised_sentence = []
        location_tokenised_sentence = []

        for index, word in enumerate(sent_words):
            if word == "$t$" or word.strip() == "":
                continue
            try:
                word_index = sent_word2idx[word]
            except:
                print("word:", word)
                print ("id not found for word in the sentence")
                exit()

            location_info = abs(index - target_location)

            if word in embeddings:
                id_tokenised_sentence.append(word_index)
                location_tokenised_sentence.append(location_info)

            # if word not in embeddings:
            #   is_included_flag = 0
            #   break

        is_included_flag = False
        for word in target_words:
            if word in embeddings:
                is_included_flag = True
                break


        try:
            target_index = target_word2idx[target]
        except:
            print("target:", target)
            print("id not found for target")
            exit()


        if not is_included_flag:
            print('not included: ',sentence)
            continue

        sentence_list.append(id_tokenised_sentence)
        location_list.append(location_tokenised_sentence)
        target_list.append(target_index)
        polarity_list.append(polarity)

    return sentence_list, location_list, target_list, polarity_list

# Till here


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
def accuracy(actuals, predictions) :
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


def load_embedding_file(embed_file_name, word_set):
    ''' loads embedding file and returns a dictionary (word -> embedding) for the words existing in the word_set '''
    print("Loading GloVe")
    embeddings = {}
    with open(embed_file_name, 'r') as embed_file:
        for line in embed_file:
            content = line.strip().split(" ")
            word = content[0]
            if word in word_set:
                embedding = np.array(content[1:], dtype=np.float32)
                embeddings[word] = embedding
    print("Loaded GloVe")
    return embeddings
