from utils import *
from memnet_model import MemN2N

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from os import path

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 15, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "./data/glove.840B.300d.txt", "pre-trained glove vectors file path [./data/glove.6B.300d.txt]")
flags.DEFINE_string("train_data", "./data/memnet_train.csv", "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("test_data", "./data/memnet_test.csv", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_integer('pad_idx', 0, 'something for the help')
flags.DEFINE_integer('nwords', 0, 'something more for the help')
flags.DEFINE_integer('mem_size', 0, 'some more shit')
flags.DEFINE_multi_float('pre_trained_context_wt', [], 'askjdbalsbd')
flags.DEFINE_multi_float('pre_trained_target_wt', [], 'askjdbalsbd')
FLAGS = flags.FLAGS


def main():
    elec_data = pd.read_csv(path.join('.', 'data', 'dl_elec_data.csv'))
    food_data = pd.read_csv(path.join('.', 'data', 'dl_food_data.csv'))
    joint_data = pd.read_csv(path.join('.', 'data', 'joint_data.csv'))

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    print(elec_data.shape)
    print(food_data.shape)

    for train_index, test_index in skf.split(elec_data, y=elec_data['class']):
        train_X, test_X = elec_data.T[train_index].T, elec_data.T[test_index].T


        source_count, target_count = [], []
        source_word2idx, target_word2idx, word_set = {}, {}, {}
        max_sent_len = -1

        max_sent_len = get_dataset_resources(train_X, source_word2idx, target_word2idx, word_set, max_sent_len)
        max_sent_len = get_dataset_resources(test_X, source_word2idx, target_word2idx, word_set, max_sent_len)
        embeddings = load_embedding_file(FLAGS.pretrain_file, word_set)

        train_data = get_dataset(train_X, source_word2idx, target_word2idx, embeddings)
        test_data = get_dataset(test_X, source_word2idx, target_word2idx, embeddings)

        print(train_data[:5])

        print("Train data size - ", len(train_data[0]))
        print("Test data size - ", len(test_data[0]))

        print("max sentence length - ", max_sent_len)
        FLAGS.pad_idx = source_word2idx['<pad>']
        FLAGS.nwords = len(source_word2idx)
        FLAGS.mem_size = max_sent_len

        FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,
                                                                                         target_word2idx, FLAGS.edim)
        with tf.Session() as sess:
            model = MemN2N(FLAGS, sess)
            model.build_model()
            model.run(train_data, test_data)

    # for train_index, test_index in skf.split(food_data, y=food_data['class']):
    #     train_X, test_X = food_data.T[train_index].T, food_data.T[test_index].T
    #     source_count, target_count = [], []
    #     source_word2idx, target_word2idx, word_set = {}, {}, {}
    #     max_sent_len = -1
    #
    #     max_sent_len = get_dataset_resources(train_X, source_word2idx, target_word2idx, word_set, max_sent_len)
    #     max_sent_len = get_dataset_resources(test_X, source_word2idx, target_word2idx, word_set, max_sent_len)
    #     embeddings = load_embedding_file(FLAGS.pretrain_file, word_set)
    #
    #     train_data = get_dataset(train_X, source_word2idx, target_word2idx, embeddings)
    #     test_data = get_dataset(test_X, source_word2idx, target_word2idx, embeddings)
    #
    #     print(train_data[:5])
    #
    #     print("Train data size - ", len(train_data[0]))
    #     print("Test data size - ", len(test_data[0]))
    #
    #     print("max sentence length - ", max_sent_len)
    #     FLAGS.pad_idx = source_word2idx['<pad>']
    #     FLAGS.nwords = len(source_word2idx)
    #     FLAGS.mem_size = max_sent_len
    #
    #     FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,
    #                                                                                      target_word2idx, FLAGS.edim)
    #     with tf.Session() as sess:
    #         model = MemN2N(FLAGS, sess)
    #         model.build_model()
    #         model.run(train_data, test_data)


if __name__ == "__main__":
    main()
