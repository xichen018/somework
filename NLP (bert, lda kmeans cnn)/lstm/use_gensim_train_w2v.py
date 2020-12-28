#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
def getContents():
    data = []
    with open('./data/train_data.txt', encoding='utf-8') as fr:
        for line in fr:
            data.append(line.strip().split()[0].split())
    return data
data = getContents()
print('get data OK. w2c train...')
model = Word2Vec(data, size=150, window=5, min_count=5)
model.save('./models/vec.model')
model.wv.save_word2vec_format('./models/vec.model.wv', binary=False)
print('w2c saved OK.')