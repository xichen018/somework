#!/usr/bin/env python
from sklearn.externals import joblib
from gensim.models import LdaModel
from gensim.corpora import Dictionary
docs = []
with open('./tmpData/cuts.txt', encoding='UTF-8') as fr:
    fr.readline()
    for line in fr:
        docs.append(line.strip().split())
print(docs)
dct = Dictionary(docs)
joblib.dump(dct, 'dct.m')
corpus = [dct.doc2bow(_) for _ in docs]
c_train, c_test = corpus[:int(len(corpus)*0.8)], corpus[int(len(corpus)*0.8):]
names = []
y = []
for i in range(10, 100, 5):
    ldamodel = LdaModel(corpus=c_train, num_topics=i, id2word=dct)
    Perword_Perplexity = ldamodel.log_perplexity(c_train) * -100
    names.append(str(i))
    y.append(Perword_Perplexity)

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
x = range(len(names))
y = y
plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'perplexity')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"topics") #X轴标签
plt.ylabel("perplexity") #Y轴标签
plt.title("perplexity plot") #标题
plt.show()
