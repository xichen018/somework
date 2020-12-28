#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
import numpy as np
class LoadVec:
    def __init__(self):
        self.model = gensim.models.Word2Vec.load('./models/vec.model')

    def getVec(self, word):
        try:
            return self.model[word]
        except Exception as e:
            return np.random.rand(300)


if __name__ == '__main__':
    lm = LoadVec()
    print(lm.getVec('爸妈'))