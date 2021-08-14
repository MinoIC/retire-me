import json
import math

import keras.regularizers
import pandas
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import shutil
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, Flatten, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')

MODEL_NAME = 'spx'
FILEPATH = './' + MODEL_NAME + '/'
GOAL = 'amzn.csv'
INPUT_LEN =60
OUTPUT_LEN = 7
PREDICT_GROUPS = 30
SCALE_MULTIPLE = 1.0
SCALE_ADD = 0.0
DIMS = ['Close', 'High', 'Low', 'Open','Volume']
Threshold = 0.002

GAP = 0
for i in range(INPUT_LEN, INPUT_LEN + OUTPUT_LEN):
    if i % OUTPUT_LEN == 0:
        GAP = i - 1


def useful(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if math.isnan(arr[i][j]) or arr[i][j] == 0.0:
                return False
    return True


def loadfile(filename):
    df = pandas.read_csv(FILEPATH + filename)
    # create a new dataframe with only the close column
    data = df.filter(DIMS)
    scaler = MinMaxScaler()
    for dim in DIMS :
        data[dim] = scaler.fit_transform(data[dim].values.reshape(-1, 1))
    dataset = data.values
    if len(dataset) <= INPUT_LEN:
        raise Exception('CSV file is too short to use')
    train_data_len = len(dataset)-(INPUT_LEN + PREDICT_GROUPS * OUTPUT_LEN)
    # Scale the data
    x_train = []
    y_train = []
    for i in range(INPUT_LEN + 1, train_data_len):
        x_train.append(np.array(dataset[i - INPUT_LEN:i, :]))
        y_train.append(np.array(dataset[i, 0]))
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], INPUT_LEN, len(DIMS)))
    return x_train, y_train, dataset[-(INPUT_LEN + PREDICT_GROUPS * OUTPUT_LEN):, :]


def build_model():
    # build the model
    model = Sequential()
    model.add(CuDNNLSTM(256, return_sequences=True,kernel_regularizer=keras.regularizers.L2(0.0006), input_shape=(INPUT_LEN, len(DIMS))))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5, activation='linear'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer="adadelta", loss='mse')
    model.summary()
    return model


# 1 means rise
# 0 means smooth
# -1 means tumble
def judge_pn(inc):
    if inc >= Threshold:
        return 1
    elif Threshold > inc > -Threshold:
        return 0
    else:
        return -1


def predict(model, test_data):
    if len(test_data) <= INPUT_LEN * 2:
        raise Exception('test data is too short to predict')
    predictions = []
    for i in range(INPUT_LEN, len(test_data) - OUTPUT_LEN):
        if i % OUTPUT_LEN == OUTPUT_LEN - 1:
            predictions.append(single_predict(model, test_data[i - INPUT_LEN:i, :]))
    predictions = np.array(predictions)
    return test_data[:, 0], predictions


def test_pn(model, test_data):
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(INPUT_LEN + 1, len(test_data) - OUTPUT_LEN):
        pred = single_predict_pn(model, test_data[i - INPUT_LEN:i, :])
        vald = judge_pn(test_data[i + OUTPUT_LEN][0] / test_data[i + 1][0] - 1.0)
        if pred == 1 and vald == 1:
            # print(pred, vald,'TP')
            TP = TP + 1.0
        if pred == 1 and vald == -1:
            # print(pred, vald,'FP')
            FP = FP + 1.0
        if pred == -1 and vald == -1:
            # print(pred, vald,'TN')
            TN = TN + 1.0
        if pred == -1 and vald == 1:
            # print(pred, vald,'FN')
            FN = FN + 1.0
    A = ((TP + TN) / (TP + TN + FP + FN)) if (TP + TN + FP + FN) != 0 else 1.0
    print(' ', A, TP, TN, FP, FN)
    return A


def single_predict(model, test_data):
    td = test_data.copy()
    x_test = td
    for i in range(OUTPUT_LEN):
        prediction = model.predict(np.reshape(x_test, (1, INPUT_LEN, len(DIMS))))
        x_test = np.append(x_test, np.reshape(np.append(prediction[0],[np.mean(x_test[:,dim]) for dim in range(1,len(DIMS))],axis=0),newshape=(1, len(DIMS))), 0)[1:]
        # inv = [[test_data[i - INPUT_LEN - 1 + j, dim] * (x_test[0, j, dim] + 1.0) for j in range(INPUT_LEN + 1)] for dim in range(len(DIMS))]
    return np.array(x_test)[-OUTPUT_LEN:, 0]


inc_count = []


def single_predict_pn(model, test_data):
    prediction = single_predict(model, test_data)
    incr = prediction[-1] / prediction[0] - 1.0
    inc_count.append(incr)
    return judge_pn(incr)


def plot_figure(test_data, predictions):
    # plot the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price')
    plt.plot(test_data, 'k')
    for i, pred in enumerate(predictions):
        padding = [None for _ in range(i * OUTPUT_LEN + GAP)]
        plt.plot(np.append(padding, pred), label='Prediction' + str(i))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("predict.png")
    plt.show()


def plot_history():
    plt.figure(figsize=(16, 8))
    # plt.subplot(2, 1, 1)
    j = json.loads(open('history.txt', 'r').read())
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(j['loss'], label='loss')
    plt.plot(j['val_loss'], label='val_loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    # plt.subplot(2, 1, 2)
    # plt.xlabel('epoch')
    # plt.ylabel('pn_accuracy')
    # plt.plot(j['pn_accuracy'], label='accuracy')
    # plt.legend()
    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    plt.savefig('history.png')
    # plt.show()


if __name__ == '__main__':
    model = build_model()
    if os.path.exists('dataset_x.npy') & os.path.exists('dataset_y.npy') & os.path.exists('dataset_test.npy'):
        x_train, y_train, test_data = np.load('dataset_x.npy'), np.load('dataset_y.npy'), np.load('dataset_test.npy')
    else:
        x_train, y_train, test_data = loadfile(GOAL)
        np.save('dataset_x', x_train)
        np.save('dataset_y', y_train)
        np.save('dataset_test', test_data)
    print("x_train.shape:",x_train.shape)
    print("y_train.shape:",y_train.shape)
    try:
        model.load_weights(MODEL_NAME + '.h5')
    except Exception as E:
        print(E)
        pn_accuracy = []
        shutil.rmtree('model_history')
        os.mkdir('model_history')
        history = model.fit(x_train, y_train, batch_size=8, epochs=100, validation_split=0.15, shuffle=True, callbacks=[
            # tf.keras.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=15, restore_best_weights=True),
            # tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: pn_accuracy.append(test_pn(model, test_data))),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: model.save_weights('model_history/' + str(epoch) + '.h5'))
        ])
        model.save_weights(MODEL_NAME + '.h5')
        #history.history['pn_accuracy'] = pn_accuracy
        with open('history.txt', 'wt') as f:
             json.dump(history.history, f, indent=4, separators=(',', ': '))
    plot_history()
    cut_test_data, predictions = predict(model, test_data)
    plot_figure(cut_test_data, predictions)
    print(test_pn(model, test_data))
    plt.figure()
    plt.plot(inc_count)
    plt.show()
