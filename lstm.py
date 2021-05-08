import json
import math
import pandas
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, Flatten, TimeDistributed
from sklearn.preprocessing import Normalizer

plt.style.use('fivethirtyeight')

FILEPATH = './dataset/'
GOAL = 'V.csv'
MODEL_NAME = 'DJI30'
INPUT_LEN = 60
OUTPUT_LEN = 4
PREDICT_GROUPS = 100
DIMS = ['Close', 'High', 'Low', 'Open']
SCALER = Normalizer()
Threshold = 0.01
GAP = 0
for i in range(INPUT_LEN, INPUT_LEN + OUTPUT_LEN):
    if i % OUTPUT_LEN == 0:
        GAP = i - 1


def loadpath():
    files = os.listdir(FILEPATH)
    x_train, y_train = [], []
    for file in files:
        try:
            ret_x, ret_y = loadfile(file)
            if len(x_train) == 0:
                x_train, y_train = ret_x, ret_y
            else:
                x_train = np.append(x_train, ret_x, axis=0)
                y_train = np.append(y_train, ret_y, axis=0)
            print(x_train.shape, y_train.shape)
        except Exception as E:
            print(E)
    return x_train, y_train


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
    dataset = data.values
    if len(dataset) <= INPUT_LEN:
        raise Exception('CSV file is too short to use')
    train_data_len = len(dataset)
    train_data = dataset[:, :]
    # Scale the data
    x_train = []
    y_train = []
    unuseful_count = 0
    for i in range(INPUT_LEN, train_data_len):
        if unuseful_count == 0 and useful(np.array(train_data[i - INPUT_LEN:i, :])):
            x_train.append(np.array(train_data[i - INPUT_LEN:i, :]))
            y_train.append(np.array(train_data[i, 0]))
        elif unuseful_count != 0:
            unuseful_count = unuseful_count - 1
        else:
            unuseful_count = INPUT_LEN
    for i in range(len(x_train)):
        y_train[i] = y_train[i] / x_train[i][0][0] - 1.0
        for dim in range(len(DIMS)):
            base = x_train[i][0][dim]
            for j in range(INPUT_LEN):
                x_train[i][j][dim] = x_train[i][j][dim] / base - 1.0
        x_train[i] = SCALER.fit_transform(x_train[i])
    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], INPUT_LEN, len(DIMS)))
    return x_train, y_train


def build_model():
    # build the model
    model = Sequential()
    model.add(CuDNNLSTM(256, return_sequences=True, input_shape=(INPUT_LEN, len(DIMS))))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5, activation='linear'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer="Adam", loss='mae', metrics=['mean_absolute_error'])
    model.summary()
    return model


def convert_to_input(test_slice):
    x_test = np.array(test_slice)
    for dim in range(len(DIMS)):
        base = x_test[0][dim]
        for j in range(INPUT_LEN):
            x_test[j][dim] = x_test[j][dim] / base - 1.0
    return np.reshape(x_test, (1, INPUT_LEN, len(DIMS)))


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
    for i in range(INPUT_LEN, len(test_data) - OUTPUT_LEN):
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
    A = (TP + TN) / (TP + TN + FP + FN)
    print(' ', A, TP, TN, FP, FN)
    return A


def single_predict(model, test_data):
    x_test = convert_to_input(test_data)
    inv = []
    for i in range(OUTPUT_LEN):
        prediction = model.predict(np.reshape(SCALER.fit_transform(x_test[0]), (1, INPUT_LEN, len(DIMS))))
        x_test = np.append(np.delete(x_test, 0, 1),
                           np.reshape([prediction[0][0].astype(np.float64)] + [np.mean(x_test[:, dim]) for dim in range(1, len(DIMS))],
                                      newshape=(1, 1, len(DIMS))), 1)
        inv = [[test_data[i - INPUT_LEN, dim] * (p + 1.0) for p in x_test[0, :, dim]] for dim in range(len(DIMS))]
        x_test = convert_to_input(np.swapaxes(np.array(inv), 0, 1))
    return inv[0][-OUTPUT_LEN:]


def single_predict_pn(model, test_data):
    prediction = single_predict(model, test_data)
    return judge_pn(prediction[-1] / prediction[0] - 1.0)


def plot_figure(test_data, predictions):
    # plot the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD')
    plt.plot(test_data, 'k')
    for i, pred in enumerate(predictions):
        padding = [None for _ in range(i * OUTPUT_LEN + GAP)]
        plt.plot(np.append(padding, pred), label='Prediction' + str(i))
    plt.savefig("predict.png")
    plt.show()


def plot_history():
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    j = json.loads(open('history.txt', 'r').read())
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(j['loss'], label='loss')
    plt.plot(j['val_loss'], label='val_loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.ylabel('pn_accuracy')
    plt.plot(j['pn_accuracy'], label='accuracy')
    plt.legend()
    plt.savefig('history.png')
    plt.show()


def load_test(filename):
    df = pandas.read_csv(FILEPATH + filename)
    # create a new dataframe with only the close column
    data = df.filter(DIMS)
    dataset = data.values
    return dataset[-(INPUT_LEN + PREDICT_GROUPS * OUTPUT_LEN):, :]


if __name__ == '__main__':
    model = build_model()
    test_data = load_test(GOAL)
    try:
        model.load_weights(MODEL_NAME + '.h5')
    except Exception as E:
        print(E)
        if os.path.exists('dataset_x.npy') & os.path.exists('dataset_y.npy'):
            x_train, y_train = np.load('dataset_x.npy'), np.load('dataset_y.npy')
        else:
            x_train, y_train = loadpath()
        np.save('dataset_x', x_train)
        np.save('dataset_y', y_train)
        print(x_train.shape, y_train.shape)
        pn_accuracy = []
        shutil.rmtree('model_history')
        os.mkdir('model_history')
        history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2, shuffle=True, callbacks=[
            # tf.keras.callbacks.EarlyStopping(mode='min', monitor='loss', patience=5, restore_best_weights=True)
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: pn_accuracy.append(test_pn(model, test_data))),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: model.save_weights('model_history/'+str(epoch)+'.h5'))
        ])
        model.save_weights(MODEL_NAME + '.h5')
        history.history['pn_accuracy'] = pn_accuracy
        with open('history.txt', 'wt') as f:
            json.dump(history.history, f, indent=4, separators=(',', ': '))
        plot_history()
    cut_test_data, predictions = predict(model, test_data)
    plot_figure(cut_test_data, predictions)
    print(test_pn(model, test_data))
