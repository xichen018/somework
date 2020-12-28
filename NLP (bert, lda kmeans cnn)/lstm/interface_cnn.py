#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import jieba
import numpy as np
from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn.externals import joblib

import logging
logging.basicConfig(
	format='%(asctime)s %(levelname)s -> %(funcName)s <%(lineno)d>: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S',
	level=logging.DEBUG
)

class LoadCnnModel:
	def __init__(self):
		self.base_dir = './models/CNN'
		self.voc_id = {}
		self.stops = []
		self.labelEncoder = ['中性', '积极', '消极']
		hdf5_file = 'weights.0001-0.7121-0.2105-0.3908-0.4267-0.7229-0.2194-0.3695-0.2276.hdf5'
		with open(os.path.join(self.base_dir, 'voc.txt'), encoding='utf-8') as fr:
			for line in fr:
				newline = line.strip().split('\t')
				self.voc_id[newline[0]] = newline[1]
		self.model = model_from_json(open(os.path.join(self.base_dir, 'model_pyramid_cnn.json'), 'r').read())
		self.model.load_weights(os.path.join(self.base_dir, hdf5_file))
		
		
		with open(os.path.join(self.base_dir, 'stopwords.txt'), encoding='utf-8') as fr:
			fr.readline()
			for line in fr:
				self.stops.append(line.strip())

	def word2id(self, contentList):
		cutx2id = []
		for cutone in contentList:
			try:
				cutx2id.append(int(self.voc_id[cutone]))
			except:
				pass
		return cutx2id

	def predict(self, line):
		word2id_num = self.word2id([word for word in jieba.cut(line.strip()) if word not in self.stops])
		vocab_processor_format = np.array(list(sequence.pad_sequences([word2id_num], maxlen=80)))
		results = self.model.predict(vocab_processor_format)
		la = results.tolist()[0].index(max(results.tolist()[0]))
		preLabel = self.labelEncoder[la]
		return str(la), preLabel

# 以下开始对新输入的句子进行模型的判断以及一级二级标签的映射
if __name__ == '__main__':

	lcm = LoadCnnModel()
	line = '平凡生活碎碎念,晚上回家阿姨告知苗小苗发烧了，去了医院验血，嗓子和肺都没事，血象有点高，开了头孢。我们等候之余还去看了月亮。'
	id_, label = lcm.predict(line)
	print(id_, label)
