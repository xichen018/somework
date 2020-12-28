#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kmeans.load_voc import LoadVec
import joblib
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
lm = LoadVec()
def getCutData():
    iniData = []
    vecData = []
    # with open('data_cut.txt') as fr:
    #     for line in fr:
    #         line_cut = line.strip().split()
    #         sentVec = 0.0
    data=pd.read_csv("news_cut.csv")
    for line in data.cut:
            line_cut=str(line).split()
            sentVec=0.0
            for word in line_cut:
                sentVec+=lm.getVec(word)
            sentVec /= len(line_cut)
            iniData.append(line)
            vecData.append(sentVec)
    return iniData, vecData


def KmeansGo(iniData, dataX, Ks=5):
    fw = open('./resCluste.txt', 'w')
    aldata = np.array(dataX)
    kmeans = KMeans(n_clusters=Ks)
    kmeans.fit(aldata)
    joblib.dump(kmeans, './models/kmeans.m')
    infos = {}
    for i, j in zip(kmeans.labels_, iniData):
        infos[i] = infos.setdefault(i, 0) + 1
        try:
            fw.write(i+'|'+''.join(j)+'\n')
        except:
            pass
    print(infos)



if __name__ == '__main__':
    iniData, vecData = getCutData()
    KmeansGo(iniData, vecData)