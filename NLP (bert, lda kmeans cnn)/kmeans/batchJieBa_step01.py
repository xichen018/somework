#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import jieba
import pandas as pd
import glob
import re


def loadStops(filename):
    stops = []
    with open(filename, encoding='utf-8') as fr:
        fr.readline()
        for line in fr:
            stops.append(line.strip())
    return stops

def cut_words(intxt):
    return " ".join([w for w in jieba.cut(re.sub(r'[0-9]+',"num",str(intxt))) if w not in stopwords])


if __name__ == "__main__":
    # Preprocesssing

    filenames = glob.glob(r"../stopwords/*")
    stopwords = []
    for filename in filenames:
        stopwords.extend(loadStops(filename))

    # fw = open('./data_cut.txt', 'w')
    # temp = pd.read_csv("../news.csv", encoding="utf-8")
    # for i in range(4,6):
    #     data= pd.DataFrame()
    #     if i<5:
    #       data=temp.iloc[i*10000:10000*(i+1),:]
    #     else:
    #       data = temp.iloc[i * 10000:, :]
    #     data.dropna(axis=0,subset=["内容"])
    #     data["cut"]= data["内容"].apply(cut_words)
    #     data.to_csv(f"news_cut_{i}.csv",encoding="utf-8")





    # for w in data["内容"]:
    #     words_lib = ' '.join([x for x in jieba.cut(w) if x not in stopwords])
    #     if words_lib.strip() != '':
    #             try:
    #                 fw.write(words_lib + '\n')
    #                 fw.flush()
    #             except:
    #                 pass





    with open('./data/result.txt', encoding='utf-8') as fr:
        for line in fr:
            line_cut = ' '.join([x for x in jieba.cut(line.strip()) if x not in stopwords if x.strip()])
            if line_cut.strip() != '':
                try:
                    fw.write(line_cut + '\n')
                    fw.flush()
                except:
                    pass
            else:
                pass
