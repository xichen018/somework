#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob
from parsepdf2txt import parse
import jieba
from sklearn.externals import joblib

def loadStops():
    stops = []
    with open('./models/stopWords.txt', encoding='utf-8') as fr:
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

stops = loadStops()

def goCuts(filenames):
    datas = []
    for filename in filenames:
        str_ = ''
        if '.txt' in filename:
            with open(filename, encoding="utf-8") as fr:
                for line in fr:
                    str_ += line.strip()
        elif '.pdf' in filename:
            str_ = parse(filename)
        else:
            pass
        seg = ' '.join([x for x in jieba.cut(str_.strip()) if x not in stops])
        datas.append(seg)
    return datas


# txt = all_path('./datas/')
# print(len(txt))
# print(len(goCuts(txt)))

class LoadModel:
    def __init__(self):
        self.ldamodel = joblib.load('./models/lda_model.m')
        self.dct = joblib.load('./models/dct.m')

    def go(self, line):
        corpus = [self.dct.doc2bow(_) for _ in [line.strip().split()]]
        doc_lda = self.ldamodel[corpus[0]]
        doc_lda_max = sorted(doc_lda, key=lambda k: k[1], reverse=True)[0]
        info = self.ldamodel.print_topic(doc_lda_max[0])
        # print(fl)
        # print('LDA:', info)
        feather = [x.strip().replace('"', '').split('*') for x in info.strip().split('+')]
        weight_feather = [float(y[0]) for y in feather]
        return info, weight_feather


    def batchPredict(self, preDir):
        filenames = all_path(preDir)
        print(filenames)
        datas = goCuts(filenames)
        fw = open('./resultsUn.txt', 'w')
        for dat, fl in zip(datas, filenames):
            print('process {}'.format(fl))
            info, weights = self.go(dat)
            #[['0.248', '价格'], ['0.051', '赠品'], ['0.045', '品质'], ['0.041', '正品'], ['0.034', '封条'], ['0.032', '物流'],['0.030', '客服'], ['0.024', '原厂'], ['0.023', '网络'], ['0.021', '老婆']]
            fw.write(fl + '|' + info + '\n')
            fw.flush()
        alldata = ' '.join(datas)
        fw.write('*****************alldata******************\n')
        info, weights = self.go(alldata)
        fw.write(info + '\n')
        fw.flush()

if __name__ == '__main__':
    lm = LoadModel()
    lm.batchPredict('./datas/')