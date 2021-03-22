import math
import pandas
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, TimeDistributed
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

FILEPATH = 'E:/wstock/SH/'
INPUTLEN = 45
OUTPUTLEN = 7
GOAL = 'SH000006.csv'
PREDICT_GROUPS = 20
GAP = 41


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
            print(x_train.shape,y_train.shape)
        except Exception as E:
            print(E)
    return x_train, y_train


def loadfile(filename):
    df = pandas.read_csv(FILEPATH + filename)
    # create a new dataframe with only the close column
    data = df.filter(['Close'])
    dataset = data.values
    if len(dataset) <= INPUTLEN:
        raise Exception('CSV file is too short to use')
    train_data_len = len(dataset)
    train_data = dataset[:, :]
    # Scale the data
    x_train = []
    y_train = []
    for i in range(INPUTLEN, train_data_len):
        x_train.append(train_data[i - INPUTLEN:i, 0])
        y_train.append(train_data[i, 0])
    x_train_scaled = []
    y_train_scaled = []
    for i in range(len(x_train)):
        scaled = [(float(p) / float(x_train[i][0]) - 1) for p in x_train[i]]
        x_train_scaled.append(scaled)
        y_train_scaled.append((float(y_train[i]) / float(x_train[i][0]) - 1.0))
    # convert to numpy arrays
    x_train, y_train = np.array(x_train_scaled), np.array(y_train_scaled)
    # reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], INPUTLEN, 1))
    return x_train, y_train


def build_model():
    # build the model
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(INPUTLEN, 1)))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5, activation='linear'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer="Adam", loss='mae', metrics=['mean_absolute_error'])
    model.summary()
    return model


def predict(model, test_data):
    if len(test_data) <= INPUTLEN * 2:
        raise Exception('test data is too short to predict')
    # create the datasets x_test and y_test
    # cut_test_data = test_data[:-(len(test_data) % INPUTLEN)]
    cut_test_data = test_data
    predictions = []
    x_test = np.array([float(p) / float(cut_test_data[0]) - 1 for p in cut_test_data[0:INPUTLEN, 0]])
    x_test = np.reshape(x_test, (1, INPUTLEN, 1))
    for i in range(INPUTLEN, len(cut_test_data)):
        prediction = model.predict(x_test)
        x_test = np.append(np.delete(x_test, 0, 1), [prediction], 1)
        inv = [cut_test_data[i - INPUTLEN] * (p + 1) for p in x_test[0, :, 0]]
        if i % OUTPUTLEN == OUTPUTLEN - 1:
            predictions.append(inv[-OUTPUTLEN:])
            x_test = np.reshape(
                [float(p) / float(cut_test_data[i - INPUTLEN + 1]) - 1 for p in
                 cut_test_data[i - INPUTLEN + 1:i + 1, 0]],
                (1, INPUTLEN, 1))
        else:
            x_test = np.reshape([float(p) / float(cut_test_data[i - INPUTLEN + 1]) - 1 for p in inv], (1, INPUTLEN, 1))
    # predictions = np.broadcast_to(predictions, (predictions.shape[0], x_train.shape[2]))
    predictions = np.array(predictions)
    # calculate the RMSE
    # rsp = np.reshape(predictions, newshape=(len(cut_test_data) - INPUTLEN))
    # vld = np.reshape(cut_test_data[INPUTLEN:], rsp.shape)
    # rmse = np.sqrt(np.mean(rsp - vld) ** 2)
    # print('rmse:', rmse)
    return cut_test_data, predictions


def plot_figure(test_data, predictions):
    # plot the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD')
    plt.plot(test_data, 'k')
    for i, pred in enumerate(predictions):
        padding = [None for _ in range(i * OUTPUTLEN + GAP)]
        plt.plot(np.append(padding, pred), label='Prediction' + str(i))
        # plt.legend()
    plt.savefig("predict.png")
    plt.show()


def load_test(filename):
    df = pandas.read_csv(FILEPATH + filename)
    # create a new dataframe with only the close column
    data = df.filter(['Close'])
    dataset = data.values
    return dataset[-(INPUTLEN+PREDICT_GROUPS*OUTPUTLEN):, :]


if __name__ == '__main__':
    model = build_model()
    try:
        model.load_weights(GOAL + '.h5')
    except:
        if os.path.exists('dataset_x.npy') & os.path.exists('dataset_y.npy'):
            x_train, y_train = np.load('dataset_x.npy'), np.load('dataset_y.npy')
        else:
            x_train, y_train = loadpath()
            np.save('dataset_x', x_train)
            np.save('dataset_y', y_train)
        print(x_train.shape, y_train.shape)
        history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2, shuffle=True, callbacks=[
            tf.keras.callbacks.EarlyStopping(mode='min', monitor='val_loss', patience=2, restore_best_weights=True)
        ])
        model.save_weights(GOAL + '.h5')
        print(history.history)
    test_data = load_test(GOAL)
    cut_test_data, predictions = predict(model, test_data)
    plot_figure(cut_test_data, predictions)
