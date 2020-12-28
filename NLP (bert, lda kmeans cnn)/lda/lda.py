#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import jieba.posseg as psg
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from parsepdf2txt import parse
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def getStopWords():
    stops = []
    with open('./modelsUN/stopWords.txt', encoding='utf-8') as fr:
        fr.readline()
        for line in fr:
            stops.append(line.strip())
    return stops


def all_path(dirname):
    result = []
    filter = ['.txt', '.pdf']
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in filter:
                result.append(apath)
    return result


def goCuts(dirs):
    stops = getStopWords()
    fw = open('./tmpData/cutsUN.txt', 'w')
    num = 0
    for filename in all_path(dirs):
        num += 1
        print('正在处理...{}'.format(filename))
        str_ = ''
        if '.txt' in filename:
            with open(filename, encoding="utf-8") as fr:
                for line in fr:
                    str_ += line.strip()
        elif '.pdf' in filename:
            str_ = parse(filename)
        else:
            pass
        seg = psg.cut(str_.strip())
        dbq = []
        for ss in seg:
            dbq.append(ss)
        pp = [x for x, y in dbq if x not in stops and y == 'n']
        if ' '.join(pp):
            fw.write(' '.join(pp) + '\n')
        fw.flush()
    fw.close()
    print('对指定文件夹下的文件构建分词完成，生成文件为：cuts.txt, 总行数为：{}'.format(num))



def goTrainLDA(n_topic=10):
    docs = []
    with open('./tmpData/cutsUN.txt') as fr:
        for line in fr:
            docs.append(line.strip().split())
    dct = Dictionary(docs)
    joblib.dump(dct, './modelsUN/dct.m')
    print('词库保存完毕！')
    corpus = [dct.doc2bow(_) for _ in docs]
    ldamodel = LdaModel(corpus=corpus, num_topics=n_topic, id2word=dct)
    joblib.dump(ldamodel, './modelsUN/lda_model.m')
    print('LDA模型保存完毕！')

if __name__ == '__main__':

    #对给定的文件夹下的内容进行：1
    goCuts("./unsupervised/ch/")

    #训练LDA 模型，并保存模型：2
    goTrainLDA(10)#参数为主题数

