#coding:utf-8
import os,logging
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

'''
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
'''
import json
from multiprocessing import Pool
import glob

def getTag(contents):
	results = []
	for xx in contents:
		try:
			indices, segments = tokenizer.encode(first=xx.strip().split('\t')[2], max_len=128)
			embedding = np.mean(model.predict([np.array([indices]), np.array([segments])])[0],0)
			#fw.write(line.strip() + '\t' + ','.join([str(x) for x in embedding])+'\n')\
			results.append(xx + '\t' + '\t' + ','.join([str(x) for x in embedding]))
		except:
			results.append(xx + '\t' + '\t' + ','.join([str(x) for x in [1]*768]))
	print(results)
	return results


def getData(fileInput):
	data = []
	with open(fileInput) as fr:
		for line in fr:
			try:
				data.append(line.strip())
			except:
				logging.error('Error..................')
	return data


def multi_process(data, num_, fw):
	finalTag = []
	# num_ = 40
	res_list = []
	pool_ = Pool(processes=num_)
	n = int(len(data) / num_)
	txtDicts = [data[i:i + n] for i in range(0, len(data), n)]

	for ii in range(0, num_):
		res = pool_.apply_async(getTag, args=(txtDicts[ii],))
		res_list.append(res)
	pool_.close()
	pool_.join()
	for res in res_list:
		# finalTag.append(res.get())
		for j in res.get():
			fw.write(j + '\n')
			fw.flush()
	fw.close()
	logging.info('All subprocessess done.')


# return finalTag
process_num = 5
import time
if __name__ == '__main__':
	for filename in glob.glob('./zl.txt'):

		logging.info('正在处理..{0}'.format(filename))
		fw = open('./%s' % filename.split('/')[-1] + '_emb.txt', 'w')
		start_ = time.time()
		data = getData(filename)
		processLineNum = len(data)
		logging.info(
			'即将处理{}条数据, 启动{}个进程，每个进程处理{}条数据，请稍等...'.format(processLineNum, process_num, processLineNum / process_num))
		multi_process(data, process_num, fw)
		end_ = time.time()
		use_time_sec = (int(end_) - int(start_))
		logging.info('总耗时：{}s,平均每条耗时{}s,转换单进程消耗时间为{}s'.format(use_time_sec, use_time_sec / processLineNum,
															  (use_time_sec / processLineNum) * process_num))
