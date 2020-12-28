#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
from gensim.models import Word2Vec
import pandas as pd
# filenames = glob.glob('./data_cut.txt')
# data = []
# for filename in filenames:
#     with open(filename) as fr:
#         for line in fr:
#             newline = line.strip()
#             data.append(newline.split())

temp=pd.read_csv("news_cut.csv",encoding="utf-8")
data=[]
for line in temp.cut:
      data.append(str(line).split())
model = Word2Vec(data, size=300,window=5, min_count=3)
model.save('./models/vec.model')
print('w2c saved OK.')