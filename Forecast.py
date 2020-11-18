import dns
import pymongo
import winsound
import msvcrt
from tqdm import tqdm
from datetime import timedelta
import time
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
import warnings
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior()
os.system('cls')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
# from tensorflow.python.framework import ops

if not sys.warnoptions:
    warnings.simplefilter('ignore')


class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias=0.1,
    ):
        def lstm_cell(size_layer):
            return tf.compat.v1.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple=False,
        )
        self.X = tf2.placeholder(tf.float32, (None, None, size))
        self.Y = tf2.placeholder(tf.float32, (None, output_size))
        drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )
        self.hidden_layer = tf2.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.compat.v1.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.compat.v1.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


def forecast(a):
    ops.reset_default_graph()
    modelnn = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    sess = tf2.InteractiveSession()
    sess.run(tf2.global_variables_initializer())
    date_ori = pd.to_datetime(df.index[:]).tolist()

    pbar = tqdm(range(epoch), desc=f'train loop {a+1}')
    for i in pbar:
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k: index, :].values, axis=0
            )
            batch_y = df_train.iloc[k + 1: index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state,
                    modelnn.optimizer, modelnn.cost],
                feed_dict={
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

    future_day = test_size

    output_predict = np.zeros(
        (df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(
                    df_train.iloc[k: k + timestamp], axis=0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1: k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(minutes=1))

    init_value = last_state

    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict={
                modelnn.X: np.expand_dims(o, axis=0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(minutes=1))

    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.4)

    return deep_future


# dataTraining = 50
simulation_size = 2
num_layers = 1
size_layer = 10
timestamp = 5
epoch = 1000
dropout_rate = 0.8
test_size = 10
learning_rate = 0.01
df = []
nomerFoto = 0

# entrada = input("Masukan Nilai Training")
start = time.time()
while (1):
    os.system('cls')
    keyBreak = 0

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    end = time.time()
    runTime = int((end - start)/60)

    while (1):
        os.system('cls')
        print("Tekan Enter untuk memulai atau tunggu dalam  beberapa menit")
        print(F'{int(keyBreak/60)}:{int(keyBreak%60)}')
        if runTime >= 0:
            print(f"Runtime = {runTime} Menit")
            print(f"Training = {len(df)} Menit")
            print(f"Prediksi = {test_size} Menit")
        keyBreak += 1
        if msvcrt.kbhit():
            if ord(msvcrt.getch()):
                break
        if keyBreak >= 60*5:
            break
        time.sleep(1)

    client = pymongo.MongoClient(
        "mongodb+srv://Forex23:7bCnUVU09Vlq6TS0@cluster0.fpun3.mongodb.net/forex?retryWrites=true&w=majority")
    db = client.forex
    collection = db.EURUSD
    data = collection.find()
    date = []
    rate = []
    df = []
    A = []
    B = []
    for X in data:
        A.append(X['EURUSD']['rate'])
        B.append(str(X['EURUSD']['date']))

    df = pd.DataFrame(A, index=B, columns=['Nilai'])
    if len(df) >= 1001:
        df = pd.DataFrame(A, index=B, columns=['Nilai'])[-600:]

    test_size = int(len(df['Nilai'])*0.1)
    if test_size >= 60:
        test_size = 60

    sns.set()
    start = time.time()
    tf.compat.v1.random.set_random_seed(1234)

    print(df)
    minmax = MinMaxScaler().fit(
        df.iloc[:, 0:1].astype('float32'))  # Close index
    df_log = minmax.transform(df.iloc[:, 0:1].astype('float32'))  # Close index
    df_log = pd.DataFrame(df_log)

    df_train = df_log
    df.shape, df_train.shape

    results = []
    for i in range(simulation_size):
        print(" ")
        print('simulation %d' % (i + 1))
        results.append(forecast(i))

    date_ori = pd.to_datetime(df.index[-60*3:]).tolist()
    for i in range(test_size):
        date_ori.append(date_ori[-1] + timedelta(minutes=1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format='%r').tolist()
    date_ori[-5:]

    accepted_results = []
    for r in results:
        if (np.array(r[-test_size:]) < np.min(df['Nilai'])).sum() == 0 and \
                (np.array(r[-test_size:]) > np.max(df['Nilai']) * 2).sum() == 0:
            accepted_results.append(r)
    len(accepted_results)

    accuracies = [calculate_accuracy(
        df['Nilai'].values, r[:-test_size]) for r in accepted_results]

    plt.figure(figsize=(20, 10))
    for no, r in enumerate(accepted_results):
        plt.plot(r[-(60*3+test_size):], label='forecast %d' % (no + 1))
    plt.plot(df['Nilai'][-60*3:], label='true trend', c='black')
    plt.legend()
    plt.title('average accuracy: %.4f' % (np.mean(accuracies)))

    x_range_future = np.arange(len(results[0][-(60*3+test_size):]))
    plt.xticks(x_range_future[::10], date_ori[::10], rotation='vertical')

    plt.savefig('Forecast 1.png')
