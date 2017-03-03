#coding:utf-8
#使用gensim生成词向量
import gensim,os 

# sentences=[['first','sentence'],['seconde','sentence']]
# model=gensim.models.Word2Vec(sentences,min_count=1)
# print model
class MySentences(object):
	def __init__(self,dirname):
		self.dirname=dirname
	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname,fname)):
				words=[]
				for word in line.split():
					words.append(word.lower())
				#words.append('yes')
				yield words

sentences=MySentences('../raw_corpus1_split')
# model=gensim.models.Word2Vec(sentences,min_count=3)
model = gensim.models.Word2Vec(sentences,size = 100,window=5,min_count=1,workers=4,sg=1, hs=0, negative=5,iter=35)
model.save('model2.m')
model.save_word2vec_format('model2.txt', binary=False)
