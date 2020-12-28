#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim
import numpy as np

DELIMITER = '\t'


def load_voc(fn, delimiter):
    voc2id = {}
    id2voc = {}
    with open(fn, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                items = l.split(delimiter)
                if not len(items) == 2:
                    print(l)
                    continue
                v, vid = items[0].strip(), items[1].strip()
                if v and vid:
                    voc2id[v] = vid
                    id2voc[vid] = v
    return voc2id, id2voc


VOC2ID, ID2VOC = load_voc('./models/voc.txt', DELIMITER)
MAX_VOCAB = len(VOC2ID.items()) + 2
print(MAX_VOCAB)
EMBEDDING_DIM = 150


class LoadVec:
    def __init__(self):
        self.model = gensim.models.Word2Vec.load('./models/vec.model')

    def getVec(self, word):
        return self.model[word]


lv = LoadVec()


def get_embedding_metrix():
    embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_DIM))
    count = 0
    for w, id in VOC2ID.items():
        try:
            assert int(id) < MAX_VOCAB
            count += 1
            embedding_matrix[int(id)] = lv.getVec(w)
        except:
            embedding_matrix[int(id)] = np.random.rand(150)

    print('#words in embedding matrix: ' + str(count))
    return embedding_matrix


EMBEDDING_MATRIX = get_embedding_metrix()


def main():
    voc2id, id2voc = load_voc('./models/voc.txt', DELIMITER)
    print(len(voc2id.items()))
    print(len(id2voc.items()))
    EMBEDDING_MATRIX = get_embedding_metrix()
    print(EMBEDDING_MATRIX)

if __name__ == '__main__':
    main()
