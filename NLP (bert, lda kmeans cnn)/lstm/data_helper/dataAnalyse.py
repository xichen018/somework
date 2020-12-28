#coding:utf-8
import jieba
import csv
def loadStops():
    stops = []
    with open('./stopwords.txt', encoding='utf-8') as fr:
        fr.readline()
        for line in fr:
            stops.append(line.strip())
    return stops
stops = loadStops()
def getLabled():
    fw = open('../data/labled_cut.txt', 'w')
    with open('../data/nCoV_train.labeled.csv', 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr)
        for line in reader:
            lab = line[-1]
            content = line[2]
            cut_data = ' '.join([x.strip() for x in jieba.cut(content) if x not in stops and x.strip() != ''])
            try:
                fw.write(cut_data + '\t' + str(lab) + '\n')
            except:
                pass

if __name__ == '__main__':
    getLabled()