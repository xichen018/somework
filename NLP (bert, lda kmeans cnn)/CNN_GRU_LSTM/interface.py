#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.preprocessing import sequence
from keras.models import model_from_json
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s -> %(funcName)s <%(lineno)d>: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

class LoadCnnModel:
    def __init__(self, vocfilename, jsonfile, hdf5file):
        self.base_dir = './'
        self.voc_id = {}
        self.stops = []
        self.labels = ['积极', '中性', '消极']
        with open(os.path.join(self.base_dir, vocfilename), encoding='utf-8') as fr:
            for line in fr:
                newline = line.strip().split('\t')
                self.voc_id[newline[0]] = newline[1]
        self.model = model_from_json(open(os.path.join(self.base_dir, jsonfile), 'r').read())
        self.model.load_weights(os.path.join(self.base_dir, hdf5file))

    def word2id(self, contentList):
        cutx2id = []
        for cutone in contentList:
            try:
                cutx2id.append(int(self.voc_id[cutone]))
            except:
                pass
        return cutx2id

    def predict(self, line):
        word2id_num = self.word2id([word for word in line.strip().split()])
        vocab_processor_format = np.array(list(sequence.pad_sequences([word2id_num], maxlen=30)))
        results = self.model.predict(vocab_processor_format)
        preLabel_ = np.argmax(results[0])
        return self.labels[preLabel_]

demo = 'Does the player and streamer work for any RTMP server?'
preList = []
realList = []
if __name__ == '__main__':
    voc_file = 'voc.txt'
    json_file = 'model_pyramid_cnn.json'
    hdf5_file = 'weights.hdf5'
    lcm = LoadCnnModel(voc_file, json_file, hdf5_file)
    pre = lcm.predict(demo)
    print(pre)
