#coding:utf-8
import os
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
class Bert_model():
    def __init__(self,pretrained_path):
        self.pretrained_path = pretrained_path
        self.config_path = os.path.join(self.pretrained_path, 'bert_config.json')
        self.checkpoint_path = os.path.join(self.pretrained_path, 'bert_model.ckpt')
        self.vocab_path = os.path.join(self.pretrained_path, 'vocab.txt')
        self.model = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,training=False,trainable=False,seq_len=128)
        token_dict = {}
        with open(self.vocab_path, 'r',encoding="utf-8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = Tokenizer(token_dict)

text = '今天 天气 不错 啊'
base_model = Bert_model('./')
model = base_model.model
tokenizer = base_model.tokenizer
fw = open('zl_embedding.txt', 'w')

fw_content = open('zl_use_content.txt', 'w')
with open('zl.txt') as fr:
	fr.readline()
	for line in fr:
		try:
			newline = line.strip().split('\t')
			indices, segments = tokenizer.encode(first=newline[2], max_len=128)
			embedding = np.mean(model.predict([np.array([indices]), np.array([segments])])[0],0)
			fw.write(line.strip() + '\t' + ','.join([str(x) for x in embedding])+'\n')

			fw_content.write(line)
		except:
			pass
