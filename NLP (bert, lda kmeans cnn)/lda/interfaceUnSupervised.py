#!/usr/bin/env python
# -*- coding: utf-8 -*-
from parsepdf2txt import parse
from sklearn.externals import joblib

class LoadModel:
    def __init__(self):
        self.ldamodel = joblib.load('./modelsUN/lda_model.m')
        print(self.ldamodel.print_topics(3))
        print(self.ldamodel.print_topic(5))
        self.dct = joblib.load('./modelsUN/dct.m')

    def batchPredict(self):
        with open('./tmpData/cuts.txt') as fr:
            for line in fr:
                corpus = [self.dct.doc2bow(_) for _ in [line.strip().split()]]
                doc_lda = self.ldamodel[corpus[0]]
                doc_lda_max = sorted(doc_lda, key=lambda k: k[1], reverse=True)[0]
                print(doc_lda)
                info = self.ldamodel.print_topic(doc_lda_max[0], 6)
                '''
                result=lda_model(test)
                print(result)
                for topic in result:
                    #print_topic(x,y) x是主题的id，y是打印该主题的前y个词，词是按权重排好序的
                    print(lda_model.print_topic(topic[0]，2))

                '''



                feather = [x.strip().replace('"', '').split('*') for x in info.strip().split('+')]
                weight_feather = [float(y[0]) for y in feather]
                word = [y[1] for y in feather]
                print('LDA:', info)
                print('word:', word)
                print('weights:', weight_feather)
                break
lm = LoadModel()
lm.batchPredict()