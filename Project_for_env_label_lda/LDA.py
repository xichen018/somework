#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import jieba.posseg as psg
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn import preprocessing
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import glob
import pandas as pd

def loadStops(filename):
    stops = []
    with open(filename, encoding='utf-8') as fr:
        fr.readline()
        for line in fr:
            stops.append(line.strip())
    return stops



def goTrainLDA(data,n_topic=10):

    dct = Dictionary(data)
    joblib.dump(dct, './models/dct.m')
    print('词库保存完毕！')
    corpus = [dct.doc2bow(_) for _ in data]
    ldamodel = LdaModel(corpus=corpus, num_topics=n_topic, id2word=dct)
    joblib.dump(ldamodel, './models/lda_model.m')
    print('LDA模型保存完毕！')

class LoadModel:
    def __init__(self):
        self.ldamodel = joblib.load('./models/lda_model.m')
        print(self.ldamodel.print_topics(3))
        print(self.ldamodel.print_topic(5))
        self.dct = joblib.load('./models/dct.m')

    def batchPredict(self,data):
            #Print the 8 topic and largest weighted words in topic
            self.ldamodel.print_topics(8, 30)

            for line in data:
                corpus = [self.dct.doc2bow(_) for _ in data]
                doc_lda = self.ldamodel[corpus[0]]
                doc_lda_max = sorted(doc_lda, key=lambda k: k[1], reverse=True)[0]
                print(doc_lda)
                info = self.ldamodel.print_topic(doc_lda_max[0], 6)



                feather = [x.strip().replace('"', '').split('*') for x in info.strip().split('+')]
                weight_feather = [float(y[0]) for y in feather]
                word = [y[1] for y in feather]
                print('LDA:', info)
                print('word:', word)
                print('weights:', weight_feather)
                self.ldamodel.print_topics(8, 30)
                break


if __name__ == '__main__':

    '''already done 
        # Preprocesssing
    
        # filenames = glob.glob(r"../stopwords/*")
        # stopwords = []
        # for filename in filenames:
        #     stopwords.extend(loadStops(filename))
    
        temp = pd.read_csv("news_cut.csv", encoding="utf-8")
        data = []
        for line in temp.cut:
            data.append(str(line).split())
    
        #训练LDA 模型，并保存
        goTrainLDA(data,8) #参数为主题数
    '''
    temp = pd.read_csv("news_cut.csv", encoding="utf-8")
    data = []
    for line in temp.cut:
            data.append(str(line).split())
    lm = LoadModel()
    lm.batchPredict(data)

