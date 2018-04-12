import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from utils import accuracy, precision, recall, f1
from os import listdir, path


embed_dim = 128
lstm_out = 196
max_features = 2000


def train(x_train, y_train, check_trained=True, batch_size=250, text="all"):
    fname = 'simple_lstm_model_'+str(batch_size)+'_'+str(embed_dim)+'_'+str(lstm_out)+'_'+text+'.h5'
    if check_trained:
        for file in listdir('./dl_models/'):
            if file == fname:
                print("Model present, returning trained model")
                return load_model(path.join('./dl_models', fname))

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=x_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=7, batch_size=batch_size, verbose=2)

    save_model(model, path.join('./dl_models', fname))

    return model


def simple_lstm(x_train, y_train, x_test, y_test, x_elec, y_elec, x_food, y_food):
    model = train(x_train, y_train)
    print(x_train.shape)
    print(x_test.shape)
    print(x_elec.shape)
    print(x_food.shape)
    # print(model.summary())

    pred_x_train = model.predict(x_train)
    pred_x_test = model.predict(x_test)
    pred_elec = model.predict(x_elec)
    pred_food = model.predict(x_food)

    precision_test = precision(np.argmax(y_test, axis=1), np.argmax(pred_x_test, axis=1), labels=(0,1,2))
    recall_test = recall(np.argmax(y_test, axis=1), np.argmax(pred_x_test, axis=1), labels=(0,1,2))
    f1_test = f1(np.argmax(y_test, axis=1), np.argmax(pred_x_test, axis=1), labels=(0,1,2))
    #
    print("Accuracy training accuracy = ", accuracy(np.argmax(y_train, axis=1), np.argmax(pred_x_train, axis=1)))
    print("Accuracy testing accuracy =", accuracy(np.argmax(y_test, axis=1), np.argmax(pred_x_test, axis=1)), "\n")
    #
    print("Precision (test data) =", precision_test)
    print("Recall (test data) =", recall_test)
    print("F1 (test data) =", f1_test, "\n")

    print("Accuracy electronics data =", accuracy(np.argmax(y_elec, axis=1), np.argmax(pred_elec, axis=1)))
    print("Accuracy food data =", accuracy(np.argmax(y_food, axis=1), np.argmax(pred_food, axis=1)), "\n")

    # Set confusion matrix for elec data and then compute precision, recall and f1_score
    precision_elec = precision(np.argmax(y_elec, axis=1), np.argmax(pred_elec, axis=1), labels=(0,1,2))
    recall_elec = recall(np.argmax(y_elec, axis=1), np.argmax(pred_elec, axis=1), labels=(0,1,2))
    f1_elec = f1(np.argmax(y_elec, axis=1), np.argmax(pred_elec, axis=1), labels=(0,1,2))

    # Set confusion matrix for food data and then compute precision, recall and f1_score
    precision_food = precision(np.argmax(y_food, axis=1), np.argmax(pred_food, axis=1), labels=(0,1,2))
    recall_food = recall(np.argmax(y_food, axis=1), np.argmax(pred_food, axis=1), labels=(0,1,2))
    f1_food = f1(np.argmax(y_food, axis=1), np.argmax(pred_food, axis=1), labels=(0,1,2))

    print("Precision (electricity data) =", precision_elec)
    print("Recall (electricity data) =", recall_elec)
    print("F1 (electricity data) =", f1_elec, "\n")

    print("Precision (food data) =", precision_food)
    print("Recall (food data) =", recall_food)
    print("F1 (food data) =", f1_food, "\n\n")


def main():
    joint_data_train = pd.read_csv('./data/ml_joint_data_train.csv')
    joint_data_test = pd.read_csv('./data/ml_joint_data_test.csv')
    elec_data = pd.read_csv('./data/ml_elec_data.csv')
    food_data = pd.read_csv('./data/ml_food_data.csv')

    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(pd.concat([elec_data, food_data], ignore_index=True)['noaspect_text'].values.astype('U'))

    joint_data = pd.concat([joint_data_train,joint_data_test,elec_data,food_data], ignore_index=True)

    total_padded_data = pad_sequences(tokenizer.texts_to_sequences(joint_data['noaspect_text'].values.astype('U')))

    x_train = total_padded_data[:joint_data_train.shape[0]]
    y_train = to_categorical(joint_data_train['class'].values.astype(np.float32), num_classes=3)
    x_test = total_padded_data[x_train.shape[0]:x_train.shape[0]+joint_data_test.shape[0]]
    y_test = to_categorical(joint_data_test['class'].values.astype(np.float32), num_classes=3)
    x_elec = total_padded_data[x_train.shape[0]+x_test.shape[0]:x_train.shape[0]+x_test.shape[0]+elec_data.shape[0]]
    y_elec = to_categorical(elec_data['class'].values.astype(np.float32), num_classes=3)
    x_food = total_padded_data[x_train.shape[0]+x_test.shape[0]+elec_data.shape[0]:]
    y_food = to_categorical(food_data['class'].values.astype(np.float32), num_classes=3)

    simple_lstm(x_train, y_train, x_test, y_test, x_elec, y_elec, x_food, y_food)


if __name__ == '__main__':
    main()



