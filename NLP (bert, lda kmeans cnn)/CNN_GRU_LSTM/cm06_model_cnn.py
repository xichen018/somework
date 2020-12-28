#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.models import Model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, Activation, BatchNormalization, Conv1D, MaxPool1D,  Input, concatenate
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import os
import load_voc as vocab
from cm06_metric import exact_match_acc, precision, recall
import logging
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

MAX_VOCAB = vocab.MAX_VOCAB
MAX_LEN = 30
EMBEDDING_DIM = vocab.EMBEDDING_DIM
MAIN_OUTPUT_DIM = 3
EMBEDDING_MATRIX = vocab.EMBEDDING_MATRIX

def load_data(fn):
    x, y = [], []
    with open(fn) as f:
        for l in f:
            if l.strip():
                items = l.strip().split('\t')
                assert len(items) == 2
                x.append(list(map(int, items[0].split())))
                yids = list(map(int, items[1].split(',')))
                ytmp = np.zeros(MAIN_OUTPUT_DIM)
                for yid in yids:
                    ytmp[yid] = 1
                y.append(ytmp)
    return x, np.asarray(y)


X_train, y_train = load_data('./data/train_data_id')
print(y_train)
X_val, y_val = load_data('./data/dev_data_id')



logging.debug(X_train[:2])
logging.debug(np.shape(X_train))
logging.debug(y_train[:2])
logging.debug(np.shape(y_train))



X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_val = sequence.pad_sequences(X_val, maxlen=MAX_LEN)
logging.debug(X_train.shape)
logging.debug(X_val.shape)

input = Input(shape=(MAX_LEN,))

embed = Embedding(MAX_VOCAB, EMBEDDING_DIM, input_length=MAX_LEN,
                  weights=[EMBEDDING_MATRIX],
                  trainable=True,
                  mask_zero=False)(input)

LAYER = 10
dropout = embed
NB_FILTERS = EMBEDDING_DIM
for i in range(LAYER):
    cnn = Conv1D(filters=NB_FILTERS,
                 kernel_size=3,
                 padding='same'
                 )(dropout)
    cnn = Conv1D(filters=NB_FILTERS,
                 kernel_size=3,
                 padding='same'
                 )(cnn)
    cnn = Activation('relu')(cnn)
    # residual = merge([cnn, dropout])
    residual = concatenate([cnn, dropout])
    pooling = MaxPool1D(strides=2, padding='same')(residual)
    bn = BatchNormalization()(pooling)
    dropout = Dropout(0.1)(bn)

dropout = GlobalAveragePooling1D()(dropout)
dense = Dense(128, activation='relu')(dropout)
bn = BatchNormalization()(dense)
dropout = Dropout(0.1)(bn)
dense = Dense(128, activation='relu')(dropout)
bn = BatchNormalization()(dense)
dropout = Dropout(0.1)(bn)

output = Dense(MAIN_OUTPUT_DIM, activation='sigmoid')(dropout)

model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[exact_match_acc, precision, recall])

logging.info(model.summary())

MODEL_BASE_DIR = './models/cnn/'
if not os.path.exists(MODEL_BASE_DIR):
    os.makedirs(MODEL_BASE_DIR)
with open(MODEL_BASE_DIR + '/model_pyramid_cnn.json', 'w') as fw:
    fw.write(model.to_json())


save_each_epoch = ModelCheckpoint(
    MODEL_BASE_DIR + '/weights.{epoch:04d}-{loss:.4f}-{exact_match_acc:.4f}-{precision:.4f}-{recall:.4f}-{val_loss:.4f}-{val_exact_match_acc:.4f}-{val_precision:.4f}-{val_recall:.4f}.hdf5',
    monitor='val_loss',
    verbose=10,
    save_best_only=False,
    mode='auto'
)

history = LossHistory()

model.fit(X_train, y_train,
          batch_size=64,
          epochs=1,
          callbacks=[history],
          validation_data=(X_val, y_val))
history.loss_plot('epoch')

pre = model.predict(X_val)
from sklearn.metrics import classification_report

def getXXX(preList):
    datas = []
    for i in preList:
        datas.append(np.argmax(i))
    return datas

print(y_val)
print(pre)
y_val = getXXX(y_val)
pre = getXXX(pre)
print(y_val)
print(pre)
print(classification_report(getXXX(y_val), getXXX(pre)))