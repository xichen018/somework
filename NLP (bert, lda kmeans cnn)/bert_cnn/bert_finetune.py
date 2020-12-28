#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

SEQ_LEN = 85
BATCH_SIZE = 64
EPOCHS = 20 
LR = 1e-3

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


file_name = 'train.txt'
def read_file(path):
    all_text = []
    file = open(path, "r",encoding="utf-8")
    for i in file.readlines():
        all_text.append(i.strip().replace("\n", ""))
    file.close()
    return all_text

'''
f = open('jtkd_level6subNone_label_dict.txt','r')
a = f.read()
dict_name = eval(a)
print(dict_name)
f.close()
'''
dict_name={
0:"ORG-AFF",
1:"PHYS",
2:"PART-WHOLE",
3:"GEN-AFF"
}
l6 = read_file('./jtkd_l6.txt')


def load_data(file_name):
    global tokenizer
    indices, sentiments ,label_arr = [], [], []
    all_text = read_file(file_name)
    for line in tqdm(all_text):
        line = line.strip().split('\t')
        text = line[0]
        ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
        indices.append(ids)
        if line[-1].strip() in l6:
            label_arr.append(line[-1])
        else:
            print(line[-1].strip())
        
    y_ = np.zeros((len(label_arr),len(dict_name)))
    new_dict = {v : k for k, v in dict_name.items()}
    for i in range(len(label_arr)):
        lab = label_arr[i].split('|')
        for l in lab:
            l = l.strip()
            y_[i][new_dict[l]] = 1
            
    items = list(zip(indices, y_))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(sentiments)

train_data = './train.txt'
test_data = 'dev.txt'
data_x, data_y = load_data(train_data)

test_x, test_y = load_data(test_data)

#from tensorflow.python import keras
import keras
from keras_bert import AdamWarmup, calc_train_steps
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
#from keras.layers import Input,SpatialDropout1D,CuDNNLSTM,Bidirectional,Conv1D,GlobalMaxPool1D,concatenate,Dropout,Dense

from keras.layers import Input,Bidirectional,Conv1D,GlobalMaxPool1D,concatenate,Dropout,Dense
early_stopping = EarlyStopping(monitor="val_acc", mode="auto", patience=10)
callbacks_list = [early_stopping]

for l in model.layers[-2:]:
    l.trainable = True
    
inputs = model.inputs[:2]

dense = model.get_layer('NSP-Dense').output
#dense = model.get_layer('Encoder-12-FeedForward-Norm').output
#print(dense.shape)
#dense = keras.layers.Conv1D(64, 3, strides=1,activation='relu')(dense)
#print(dense.shape)
#dense = keras.layers.Flatten()(dense)
outputs = keras.layers.Dense(units=len(dict_name), activation='softmax')(Dropout(0.7)(dense))



model = keras.models.Model(inputs, outputs)

adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
model.fit(data_x, data_y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2,
          shuffle = True,
	  validation_data=(test_x, test_y),
          callbacks=callbacks_list)
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=1992)
# for train_index, test_index in skf.split(data_x, data_y.argmax(1)):
#     kfold_X_train,kfold_X_valid = train_query_arr[train_index],train_query_arr[test_index]
#     kfold_y_train,kfold_y_valid = y_[train_index], y_[test_index]
#     model.fit(kfold_X_train, kfold_y_train,
#               validation_data=(kfold_X_valid,kfold_y_valid),
#               batch_size=BATCH_SIZE,
#               epochs=EPOCHS, 
#               shuffle = True)
y_pred = model.predict(test_x)
preds = []
reals = []
for j, k in zip(y_pred, test_y):
    preds.append(np.argmax(j))
    reals.append(np.argmax(k))
from sklearn.metrics import classification_report
print(classification_report(preds, reals))
model.save_weights('model.h5') 
