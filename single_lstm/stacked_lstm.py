#coding:utf-8
# 单通道的LSTM 七情绪分类
from __future__ import division
import gensim
import random
import time
import numpy as np  # keras每次产生确定的数据
np.random.seed(1333)

# seed=7777
# random.seed(seed)  # trains 每次产生都一样

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

ISOTIMEFORMAT='%Y-%m-%d %X'
maxlen=120
nb_classes=7

batch_size=32
embedding_dim=100
lstm_output_dim=128
hidden_dim=64

class Document:
	def __init__(self,polarity,words):
		self.polarity=polarity
		self.words=words

def readFromFile(path,polarity):
	documents=[]
	with open(path,'r') as f:
		for line in f:
			pieces=line.strip().split()
			words=[]
			for piece in pieces:
				word=piece.lower()
				#words[word]=words.get(word,0)+1
				words.append(word)
			documents.append(Document(polarity,words))
	return documents

def Domain():
	em0=readFromFile('raw_corpus1_split/resort_anger.txt',0)
	em1=readFromFile('raw_corpus1_split/resort_disgust.txt',1)
	em2=readFromFile('raw_corpus1_split/resort_fear.txt',2)
	em3=readFromFile('raw_corpus1_split/resort_happiness.txt',3)
	em4=readFromFile('raw_corpus1_split/resort_like.txt',4)
	em5=readFromFile('raw_corpus1_split/resort_sadness.txt',5)
	em6=readFromFile('raw_corpus1_split/resort_surprise.txt',6)
	return em0,em1,em2,em3,em4,em5,em6

def createVec(docs,my_model): # 生成样本的词向量
	train_vec=[]
	train_label=[]
	for doc in docs: 
		sen=[]
		for word in doc.words:
			if word in my_model:
				sen.append(my_model[word])
		t=len(sen)
		if t==0:
			if len(train_vec)==0:
				s=[]
				for i in range(120):
					s.append(random.random())
				sen.append(s)
				t=1
			else:
				train_vec.append(train_vec[0])
				train_label.append(doc.polarity)
				continue
		if t<maxlen:
			index=0
			while len(sen)<maxlen:
				sen.append(sen[index])
				index+=1
		else:
			sen=sen[:maxlen]
		train_vec.append(sen)
		train_label.append(doc.polarity)
	return train_vec,train_label

def load_data(my_model):
	em0,em1,em2,em3,em4,em5,em6=Domain()
	print 'loading %s data...' %(my_model)
	# train=em0[90:450]+em1[90:450]+em2[90:450]+em3[18:90]+em4[90:450]+em5[90:450]+em6[90:450]
	test=em0[:72]+em1[:72]+em2[:72]+em3[:72]+em4[:72]+em5[:72]+em6[:72]   # 前90(18)个作为测试样本
	val=em0[-30:]+em1[-30:]+em2[-30:]+em3[-30:]+em4[-30:]+em5[-30:]+em6[-30:]  #  验证集 后60个
	em2_new=[]
	for i in range(360-len(em2)):
		em2.append(random.choice(em2[72:-30]))  # re-sampling 扩充到 360

	train=random.sample(em0[72:-30],258)+random.sample(em1[72:-30],258)+random.sample(em2[72:-30],258)+\
	      random.sample(em3[72:-30],258)+random.sample(em4[72:-30],258)+random.sample(em5[72:-30],258)+\
	      random.sample(em6[72:-30],258)

	# val=random.sample(em0[228:],60)+random.sample(em1[228:],60)+random.sample(em2[228:],60)+\
	#       random.sample(em3[228:],60)+random.sample(em4[228:],60)+random.sample(em5[228:],60)+\
	#       random.sample(em6[228:],60)
	
	# my_model=gensim.models.Word2Vec.load(model_path)  # 词向量模型
	# print 'type of my_model:'+str(type(my_model))

	train_vec,train_label=createVec(train,my_model)   #  生成词向量
	val_vec,val_label=createVec(val,my_model)
	test_vec,test_label=createVec(test,my_model)
	
	X_train=np.array(train_vec)  # 将样本转换为array形式
	X_val=np.array(val_vec)
	X_test=np.array(test_vec)

	Y_train=np_utils.to_categorical(train_label,nb_classes)  # 将类别列表转为二进制矩阵
	Y_val=np_utils.to_categorical(val_label,nb_classes)
	Y_test=np_utils.to_categorical(test_label,nb_classes)

	return X_train,X_test,Y_train,Y_test,X_val,Y_val,test_label

# def LSTM_model(X_train,X_test,Y_train,Y_test,test_label):   # 生成LSTM网络
	
# 	print('Loading embedding successful!')
# 	print('len(X_train):'+str(len(X_train)))
# 	print('len(X_test):'+str(len(X_test)))
# 	print('len(Y_train):'+str(len(Y_train)))
# 	print('len(Y_test):'+str(len(Y_test)))
# 	# print(test_label)
# 	print('X_train shape:',X_train.shape)
# 	print('X_test shape:',X_test.shape)
# 	print('Build model...')

# 	model=Sequential()

# 	model.add(LSTM(lstm_output_dim,input_shape=(maxlen,embedding_dim)))  # LSTM 层 100->128

# 	model.add(Dense(hidden_dim))   # 隐藏层 全连接层  128->64
# 	model.add(Activation('relu'))
# 	model.add(Dropout(0.5))

# 	model.add(Dense(nb_classes))
# 	model.add(Activation('softmax'))

# 	model.compile(loss='categorical_crossentropy', optimizer='adam')
# 	hist=model.fit(X_train,Y_train, batch_size=20, nb_epoch=20, verbose=1, shuffle=True,show_accuracy=True,   #20 10
# 	                validation_data=(X_test,Y_test))
# 	print (hist.history) 
# 	best_acc=max(hist.history['val_acc'])
# 	print('the best epoch:%d,and the acc:%.5f' %(hist.history['val_acc'].index(best_acc)+1,best_acc))
nb_epoch=30
def LSTM_model(X_train,X_test,Y_train,Y_test,test_label,X_val,Y_val):   # 生成LSTM网络
	
	print('Loading embedding successful!')
	print('len(X_train):'+str(len(X_train)))
	print('len(X_val):'+str(len(X_val)))
	print('len(X_test):'+str(len(X_test)))
	print('len(Y_train):'+str(len(Y_train)))
	print('len(Y_val):')+str(len(Y_val))
	print('len(Y_test):'+str(len(Y_test)))
	# print(test_label)
	print('X_train shape:',X_train.shape)
	print('X_test shape:',X_test.shape)
	print('Build model...')

	model=Sequential()  # stacked lstm model
	model.add(LSTM(hidden_dim,return_sequences=True, 
				input_shape=(maxlen,embedding_dim)))
	model.add(LSTM(hidden_dim,return_sequences=True))
	model.add(LSTM(hidden_dim))
	model.add(Dense(nb_classes,activation='softmax'))
	# model.add(LSTM(lstm_output_dim,input_shape=(maxlen,embedding_dim)))  # LSTM 层 100->128

	# model.add(Dense(hidden_dim))   # 隐藏层 全连接层  128->64
	# model.add(Activation('relu'))
	# model.add(Dropout(0.5))

	# model.add(Dense(nb_classes))
	# model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')

	best_acc=0
	best_pred_label=[]
	best_pred_prob=[]
	best_epoch=0
	a0=0
	b0=0
	for i in range(nb_epoch):
		print('%d epoch...' %(i+1))
		print('Now:best_acc:%.5f \t best_epoch:%d' %(best_acc,best_epoch))
		hist=model.fit(X_train,Y_train, batch_size=32, nb_epoch=1, verbose=1, shuffle=True,show_accuracy=True,   #20 10
	                validation_data=(X_val,Y_val))
		acc=max(hist.history['val_acc'])
		p_label=model.predict_classes(X_test,batch_size=32,verbose=1)
		p_prob=model.predict_proba(X_test,batch_size=32,verbose=1)
		print p_label
		a1=np_utils.accuracy(p_label,test_label)
		print 'Now epoch test acc:%.5f' %(a1)
		if a1>a0:
			print 'a1 better:%.5f' %(a1)
			a0=a1
			b0=(i+1)
		if acc>best_acc:
			print('出现更好的acc,正在更新acc和epoch...')
			best_acc=acc
			best_pred_label=p_label
			best_pred_prob=p_prob
			best_epoch=(i+1)
	test_acc=np_utils.accuracy(best_pred_label,test_label)  # 得到正确率
	print('the best epoch:%d,and the acc:%.5f.,while best test acc epoch:%d,%.5f' %(best_epoch,test_acc,b0,a0))
	# print('the best pred_class:\n')
	# print(best_pred_label)
	write2File(best_pred_label,test_label)
	return best_pred_label,best_epoch,best_pred_prob

def write2File(pred_label,real_label):  # 将真实类别和预测结果写入文件保存
	f=open('single_result.txt','w')
	for i in range(len(pred_label)):
		f.write('%d\t%d\n' %(real_label[i],pred_label[i]))
	f.close()

def createResult(pred_label,real_label,seed,best_epoch):  # 计算prf值 写入文件
	# pass
	# tp0=tp1=tp2=tp3=tp4=tp5=tp6=0   # 
    # fp0=fp1=fp2=fp3=fp4=fp5=fp6=0
    accCount=0
    # p1=p2=p3=p4=p5=p6=p7=0
    p=[0]*7
    tp=[0]*7
    fp=[0]*7
    for i in range(len(pred_label)):
    	t_label=real_label[i]   # 真实类别
    	p_label=pred_label[i]   # 预测类别

    	for index in range(7):  
    		if t_label==index:  # t_label 等于当前类别index
    			p[index]+=1     # index类别数+1
    			if p_label==t_label:  # 预测类别==真实类别
    				tp[p_label]+=1   
    				accCount+=1
    			else:            # 预测类别不等于真实类别
    				fp[p_label]+=1
    acc=accCount/len(real_label)   # 正确率

    precision=[0]*7
    recall=[0]*7
    F1=[0]*7
    dot=1.0
    nowTime=time.strftime(ISOTIMEFORMAT,time.localtime())
    f=open('single_prf.txt','a')
    f.write('--------epoch:%d---seed:%d---time:%s--------\n' %(nb_epoch,seed,nowTime))
    for i in range(7):
    	precision[i]+=(tp[i]/(tp[i]+fp[i]))
    	recall[i]+=(tp[i]/p[i])
    	dot*=recall[i]
    	F1[i]+=((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    	f.write('label:%d \t precision:%.5f \t recall:%.5f \t F1:%.5f\n' %(i,precision[i],recall[i],F1[i]))
    G_mean=dot**(1.0/7.0)
    f.write('-----acc:%.5f---G_mean:%.5f---best_epoch:%d------\n\n' %(acc,G_mean,best_epoch))
    f.close()
    return acc,G_mean

def run(seed):
	em0,em1,em2,em3,em4,em5,em6=Domain()
	print('len(em0):'+str(len(em0)))  # 1038
	print('len(em1):'+str(len(em1)))   # 472
	print('len(em2):'+str(len(em2)))   # 581
	print('len(em3):'+str(len(em3)))   # 94
	print('len(em4):'+str(len(em4)))   # 1179
	print('len(em5):'+str(len(em5)))   # 1131
	print('len(em6):'+str(len(em6)))   # 2175
	my_model=gensim.models.Word2Vec.load(r'Model/model2.m')  # 词向量模型
	print 'type of my_model:'+str(type(my_model))

	X_train,X_test,Y_train,Y_test,X_val,Y_val,test_label=load_data(my_model)
	best_pred_label,best_epoch,best_pred_prob = LSTM_model(X_train,X_test,Y_train,Y_test,test_label,X_val,Y_val)
	for item in best_pred_prob:
		print item
	acc,G_mean=createResult(best_pred_label,test_label,seed,best_epoch)
	return acc,G_mean

if __name__ == '__main__':
	seed=[1,1111,2222,3333,4444,6666,7777,8888,9999]
	# seed=[1111]
	accList=[]
	sum_acc=0
	sum_gmean=0
	for item in seed:
		random.seed(item)
		print('seed:',item)
		acc,G_mean=run(item)
		sum_acc+=acc
		sum_gmean+=G_mean
		accList.append(acc)
	for i in range(len(seed)):
		print('%d:%.5f' %(seed[i],accList[i]))
	ave_acc=sum_acc/len(seed)
	ave_gmean=sum_gmean/len(seed)
	print 'ave_acc:%.5f' %(ave_acc)
	print 'ave_gmean:%.5f' %(ave_gmean)


	