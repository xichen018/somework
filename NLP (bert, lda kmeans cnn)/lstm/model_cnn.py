#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, \
    Dropout, Activation, BatchNormalization, Conv1D, Flatten, MaxPool1D, LSTM, \
    TimeDistributed, RepeatVector, Permute, Input, merge, Lambda, Bidirectional, \
    concatenate
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

import load_voc as vocab
from cm06_metric import exact_match_acc, precision, recall

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


MAX_VOCAB = vocab.MAX_VOCAB
MAX_LEN = 80
EMBEDDING_DIM = vocab.EMBEDDING_DIM
MAIN_OUTPUT_DIM = 3
EMBEDDING_MATRIX = vocab.EMBEDDING_MATRIX

def load_data(fn):
    x, y = [], []
    with open(fn, 'rb') as f:
        for l in f:
            l=l.decode()
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
MODEL_BASE_DIR = './models/CNN'
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
model.fit(X_train, y_train,
          batch_size=128,
          epochs=5,
          validation_data=(X_val, y_val),
          callbacks=[save_each_epoch])

import numpy as np
from sklearn.metrics import classification_report
y_true = []
y_pred = []
for i, j in zip(model.predict(X_val), y_val):
    y_true.append(np.argmax(j))
    y_pred.append(np.argmax(i))
print(classification_report(y_true, y_pred))