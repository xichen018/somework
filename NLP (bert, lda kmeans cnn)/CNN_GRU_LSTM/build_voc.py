#!/usr/bin/env python
# -*- coding: utf-8 -*-
DELIMITER = '|'
def get_character_vocabulary(fn):
    with open(fn, encoding='utf-8') as f:
        voc = []
        for l in f:
            if l.strip():
                items = l.strip().split(DELIMITER)
                try:
                    assert len(items) == 2
                    content = items[0]
                    voc += [x for x in content.split() if len(x) > 0]
                except:
                    print('nono')
    return list(set(voc))


def store_voc(voc, fwn):
    with open(fwn, 'wb') as fw:
        for id, v in enumerate(sorted(voc)):
            if v != ' ':
                fw.write((str(v) + '\t' + str(id) + '\n').encode(encoding='utf_8', errors='strict'))

def main():
    voc = get_character_vocabulary('train_data_label')
    print(len(voc))
    store_voc(voc, './models/voc.txt')


if __name__ == '__main__':
    main()
