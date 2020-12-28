#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Conv2D, Input, MaxPooling1D, Conv1D,concatenate, BatchNormalization,GlobalMaxPool1D
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import os
import load_voc as vocab
from cm06_metric import exact_match_acc, precision, recall
import logging
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
import load_voc as vob
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import pandas as pd
import tensorflow as tf
from six.moves import xrange
from tqdm import tqdm
from keras.layers import Bidirectional
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


vocab_dim = 150  # 向量维度
maxlen = 30  # 文本保留的最大长度
batch_size = 64
n_epoch = 10
input_length = 30
EMBEDDING_MATRIX = vocab.EMBEDDING_MATRIX
n_symbols = vocab.MAX_VOCAB
print(n_symbols)
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


import matplotlib.pyplot as plt
import keras
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



# 定义网络结构

MODEL_BASE_DIR = './models/models_cnn2bilstm/'


save_each_epoch = ModelCheckpoint(
    MODEL_BASE_DIR + '/weights.{epoch:04d}-{loss:.4f}-{acc:.4f}.hdf5',
    monitor='acc',
    verbose=10,
    save_best_only=False,
    mode='auto'
)

def train_cnn_bilstm(p_X_train, p_y_train, p_X_test, p_y_test):
    print(u'创建模型...')
    input = Input(shape=(MAX_LEN,))

    embed = Embedding(MAX_VOCAB, EMBEDDING_DIM, input_length=MAX_LEN,
                      weights=[EMBEDDING_MATRIX],
                      trainable=True,
                      mask_zero=False)(input)

    cov1 = Conv1D(filters=512,
                 kernel_size=3,
                 padding='same'
                 )(embed)
    max_pool1 = Activation('relu')(cov1)
    max_pool1 = GlobalMaxPool1D()(max_pool1)

    lst = LSTM(units=1024, return_sequences=True)(embed)
    max_pool2 = Activation('relu')(lst)
    dropout = Dropout(0.1)(max_pool2)
    max_pool2 = GlobalMaxPool1D()(dropout)

    merge1 = concatenate([max_pool1, max_pool2], axis=1)
    merge1 = Dropout(0.1)(merge1)
    # pooling = MaxPooling1D(strides=2, padding='same')(merge1)
    bn = BatchNormalization()(merge1)
    dropout = Dropout(0.1)(bn)
    output = Dense(MAIN_OUTPUT_DIM, activation='relu')(Dropout(0.5)(dropout))

    model = Model(input, output)
    print(model.summary())
    print(u'编译模型...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(u"训练...")
    history = LossHistory()
    with open(MODEL_BASE_DIR + '/model_pyramid_cnn.json', 'w') as fw:
        fw.write(model.to_json())
    model.fit(p_X_train, p_y_train, batch_size=batch_size, nb_epoch=n_epoch, callbacks=[history],
              validation_data=(p_X_test, p_y_test))

    print(u"评估...")
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
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
	
	

train_cnn_bilstm(X_train, y_train, X_val, y_val)


