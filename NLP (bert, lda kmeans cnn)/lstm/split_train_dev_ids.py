#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
def readtag():
    tags2id = {}
    with open('./models/voc.txt', encoding='utf-8') as fr:
        for line in fr:
            newline = line.strip().split('\t')
            tags2id[newline[0]] = newline[1]
    return tags2id

def tag2id(filename, fwfile):
    tags2id = readtag()
    fw = open(fwfile, 'w')
    with open(filename, encoding='utf-8') as fr:
        for line in fr:
            if line.strip():
                tmp = []
                newline = line.strip().split('\t')
                try:
                    assert len(newline) == 2
                    for x in newline[0].split():
                        try:
                            tmp.append(tags2id[x])
                        except:
                            pass
                    fw.write(' '.join(tmp) + '\t' + str(int(newline[-1].strip()) - 1) + '\n')
                except Exception as e:
                    pass
tag2id('./data/train_data.txt', './data/cut_datas_ids')#将全部训练数据转化为id标识，label也进行id标识

#以下开始训练集数据集的划分
def run_split_train_dev(filenaem_train):
    X = []
    y = []
    fw_train = open('./data/train_data_id', 'w')
    fw_dev = open('./data/dev_data_id', 'w')
    with open(filenaem_train) as fr:
        for line in fr:
            if line.strip():
                newline = line.strip().split('\t')
                try:
                    if len(newline) == 2:
                        y.append(','.join(list(set(newline[1].split(',')))))
                        X.append(newline[0])
                    else:
                        pass
                except:
                    pass
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, shuffle=True)
    for x, y in zip(X_train,y_train):
        fw_train.write(x + '\t' + y + '\n')
    for x, y in zip(X_test,y_test):
        fw_dev.write(x + '\t' + y + '\n')

run_split_train_dev('./data/cut_datas_ids')#训练集数据的划分，并将数据随机打乱
