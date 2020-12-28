
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf



os.environ["CUDA_VISIBLE_DEVICES"] = "3"

SEQ_LEN = 512
BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-3
LABEL_NUM=2

pretrained_path = './'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=False,
    seq_len=SEQ_LEN,
)

token_dict = {}
with codecs.open('./vocab.txt', 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)



def load_data(data):

    indices= []

    for line in tqdm(data["内容"]):
        text = str(line).strip().split('\t')

        ids, segments = tokenizer.encode(text[0], max_len=SEQ_LEN)
        indices.append(ids)
    label_arr=list(data.label)
    y_ = np.zeros((len(label_arr), LABEL_NUM))
    for i,label in enumerate(label_arr):
        if label==7:
            y_[i][1]=1
        else:
            y_[i][0]=1


    items = list(zip(indices, y_))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(sentiments)

def load_data_raw(data):
    global tokenizer
    indices= []

    for line in tqdm(data["内容"]):
        text = str(line).strip().split('\t')

        ids, segments = tokenizer.encode(text[0], max_len=SEQ_LEN)
        indices.append(ids)

    return [indices, np.zeros_like(indices)]

raw = pd.read_csv("news_cut.csv",encoding="utf-8")
raw_x = load_data_raw(raw)

data = pd.read_csv("LABEL.csv",encoding="utf-8")
data_x, data_y = load_data(data)
x_train, x_test=data_x[:700], data_x[700:]
y_train, y_test=data_y[:700], data_y[700:]



# from tensorflow.python import keras
import keras
from keras_bert import AdamWarmup, calc_train_steps
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.layers import Input,SpatialDropout1D,CuDNNLSTM,Bidirectional,Conv1D,GlobalMaxPool1D,concatenate,Dropout,Dense

from keras.layers import Input, Bidirectional, Conv1D, GlobalMaxPool1D, concatenate, Dropout, Dense

early_stopping = EarlyStopping(monitor="val_accuracy", mode="auto", patience=10)
callbacks_list = [early_stopping]

for l in model.layers[-2:]:
    l.trainable = True

inputs = model.inputs[:2]

dense = model.get_layer('NSP-Dense').output
# dense = model.get_layer('Encoder-12-FeedForward-Norm').output
# print(dense.shape)
# dense = keras.layers.Conv1D(64, 3, strides=1,activation='relu')(dense)
# print(dense.shape)
# dense = keras.layers.Flatten()(dense)
outputs = keras.layers.Dense(units=LABEL_NUM, activation='softmax')(Dropout(0.7)(dense))

model = keras.models.Model(inputs, outputs)

adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
model.fit(data_x, data_y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2,
          shuffle=True,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)

# y_pred = model.predict(x_test)
y_pre = model.predict(raw_x)
raw["label"]= y_pre
raw.to_csv("predict_news")

# preds = []
# reals = []
# for j, k in zip(y_pred, y_test):
#     preds.append(np.argmax(j))
#     reals.append(np.argmax(k))
# from sklearn.metrics import classification_report
#
# print(classification_report(preds, reals))
# model.save('model.h5')
