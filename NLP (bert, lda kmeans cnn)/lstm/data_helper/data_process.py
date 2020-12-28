#coding=utf-8
import random
if __name__ == '__main__':
    infos = {}
    with open('../data/labled_cut.txt') as fr:
        fr.readline()
        for line in fr:
            newline = line.strip().split('\t')
            infos.setdefault(newline[-1], []).append(line.strip())
    fw = open('../data/train_data.txt', 'w', encoding='utf-8')
    datas = []
    for k, v in infos.items():
        datas.extend(random.sample(v, 4000))
    random.shuffle(datas)
    for dat in datas:
        fw.write(dat + '\n')
        fw.flush()
    fw.close()