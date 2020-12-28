#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
from kmeans.load_voc import LoadVec
import glob
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
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



def pictureK(dataX, Ks=15):
    K = range(1, Ks)
    mean_distortions = []
    aldata = np.array(dataX)
    for k in K:
        print('K= {}'.format(k))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(aldata)
        mean_distortions.append(
            sum(
                np.min(
                    cdist(aldata, kmeans.cluster_centers_, metric='euclidean'), axis=1))
            / aldata.shape[0])
    plt.plot(K, mean_distortions, 'bx-')
    plt.xlabel('k')
    font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc', size=10)
    plt.ylabel(u'平均畸变程度', fontproperties=font)
    plt.title(u'用肘部法确定最佳的K值', fontproperties=font)
    # plt.show()
    plt.savefig('./k.png')
    plt.show()
    print('save ok.')

if __name__ == '__main__':
    iniData, vecData = getCutData()
    pictureK(vecData)