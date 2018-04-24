import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


def train(x_train, y_train, check_trained=True, batch_size=250, text="all", dataset="default"):
    fname = dataset+'_simple_lstm_model_'+str(batch_size)+'_'+str(embed_dim)+'_'+str(lstm_out)+'_'+text+'.h5'
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


def simple_lstm(x_train, y_train, x_test, y_test, dataset="default"):
    model = train(x_train, y_train, dataset=dataset)
    print(x_train.shape)
    print(x_test.shape)
    # print(model.summary())

    pred_x_train = model.predict(x_train)
    pred_x_test = model.predict(x_test)

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


def main():
    elec_data = pd.read_csv('./data/ml_elec_data.csv')
    food_data = pd.read_csv('./data/ml_food_data.csv')

    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(pd.concat([elec_data, food_data], ignore_index=True)['noaspect_text'].values.astype('U'))

    total_padded_data = pad_sequences(tokenizer.texts_to_sequences(food_data['noaspect_text'].values.astype('U')))

    x_electrain, x_electest, y_electrain, y_electest = train_test_split(total_padded_data, food_data['class'])

    y_electrain = to_categorical(y_electrain.values.astype(np.float32), num_classes=3)
    y_electest = to_categorical(y_electest.values.astype(np.float32), num_classes=3)

    simple_lstm(x_electrain, y_electrain, x_electest, y_electest, dataset="food")


if __name__ == '__main__':
    main()



