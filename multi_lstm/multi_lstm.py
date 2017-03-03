#coding:utf-8
# 多通道的LSTM 七情绪分类
from __future__ import division
import gensim
import random
import time
import numpy as np  # keras每次产生确定的数据
np.random.seed(1333)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Merge
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

ISOTIMEFORMAT='%Y-%m-%d %X'
seed=8888
random.seed(seed)

maxlen=120
nb_classes=7

batch_size=32
embedding_dim=100
lstm_output_dim=128   # 128


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

# def getLexicon(documents):
# 	words={}
# 	for document in documents:
# 		for word in document.words:
# 			words[word]=words.get(word,0)+1
# 	sortedWords=sorted(words.iteritems(),key=lambda x:x[1],reverse=True)
# 	lexicon={}
# 	for i,item in enumerate(sortedWords):
# 		lexicon[item[0]]=i+1
# 	return lexicon

# documents=readFromFile('corpus/reorder/fenci/1.txt',1)
# print len(documents)
# print len(documents[0].words)
# for w in documents[0].words:
# 	print w
#######################################
# docs=em0+em1+em2+em3+em4+em5+em6
# lexicon=getLexicon(docs)
# print len(lexicon)
######################################
# def createIndex(lexicon,em,polarity):
# 	vec=[]
# 	for document in em:
# 		v=[]
# 		for word in document.words:
# 			if word in lexicon:
# 				v.append(lexicon[word])
# 		vec.append(Document(polarity,v))
# 	return vec

# vec0=createIndex(lexicon,em0,0)
# vec1=createIndex(lexicon,em1,1)
# vec2=createIndex(lexicon,em2,2)
# vec3=createIndex(lexicon,em3,3)
# vec4=createIndex(lexicon,em4,4)
# vec5=createIndex(lexicon,em5,5)
# vec6=createIndex(lexicon,em6,6)

# print('len(vec0):'+str(len(vec0)))
# print('len(vec1):'+str(len(vec1)))
# print('len(vec2):'+str(len(vec2)))
# print('len(vec3):'+str(len(vec3)))
# print('len(vec4):'+str(len(vec4)))
# print('len(vec5):'+str(len(vec5)))
# print('len(vec6):'+str(len(vec6)))

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
				for i in range(100):
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
	# em0,em1,em2,em3,em4,em5,em6=Domain()
	# train=em0[90:450]+em1[90:450]+em2[90:450]+em3[18:90]+em4[90:450]+em5[90:450]+em6[90:450]
	test=em0[:72]+em1[:72]+em2[:72]+em3[:72]+em4[:72]+em5[:72]+em6[:72]   # 前72(18)个作为测试样本
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

	Y_train=np_utils.to_categorical(train_label,nb_classes)  # 将类别列表转为二进制矩阵 binary arrays
	Y_val=np_utils.to_categorical(val_label,nb_classes)
	Y_test=np_utils.to_categorical(test_label,nb_classes)

	return X_train,X_test,Y_train,Y_test,X_val,Y_val,test_label

def LSTM_model(X_train,X_test,Y_train,Y_test,test_label):   # 生成LSTM层
	
	print('Loading embedding successful!')
	print('len(X_train):'+str(len(X_train)))
	print('len(X_test):'+str(len(X_test)))
	print('len(Y_train):'+str(len(Y_train)))
	print('len(Y_test):'+str(len(Y_test)))
	# print(test_label)
	print('X_train shape:',X_train.shape)
	print('X_test shape:',X_test.shape)
	print('Build model...')

	model=Sequential()

	model.add(LSTM(lstm_output_dim,input_shape=(maxlen,embedding_dim)))  # LSTM 层 120->128

	model.add(Dense(hidden_dim))   # 隐藏层 全连接层  128->64
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	return model
	'''
	errCount=0
	for i,item in enumerate(test_label):
		if pred_label[i]!=test_label[i]:
			errCount+=1
			print('%d err occured! pred:%s,real:%s' %(errCount,pred_label[i],test_label[i]))
	print('the predict acc:%.5f' %((len(test_label)-errCount)/len(test_label)))
	'''

def getAllModels(my_model,em0,em1,em2,em3,em4,em5,em6):  # 生成5个channel
	X_train=[]
	X_test=[]
	Y_train=[]
	Y_test=[]
	X_val=[]
	Y_val=[]
	test_label=[]
	models_all=[]
	# for i in range(360-len(em2)):
	# 	em2.append(random.choice(em2))  # re-sampling 扩充到 450
	count=5  # 通道个数
	for i in range(count):
		item=load_data(my_model)  # 载入数据
		# random.seed(1)
		# random.shuffle(item[0])
		X_train.append(item[0])

		X_test.append(item[1])

		# random.seed(1)
		# random.shuffle(item[2])
		Y_train.append(item[2])

		Y_test.append(item[3])
		X_val.append(item[4])
		Y_val.append(item[5])
		test_label.append(item[6])
		m=LSTM_model(item[0],item[1],item[2],item[3],item[6])  # 建立一个LSTM模型
		models_all.append(m)     # 加入模型列表
	print('%d个样本建立完毕...' %count)
	return models_all,X_train,X_test,Y_train,Y_test,test_label,X_val,Y_val

# def Multi_channel(models_all,X_train,X_test,Y_train,Y_test,test_label):  # multi-channel 的神经网络
# 	print('开始Merge所有的通道...')
# 	model=Sequential()
# 	model.add(Merge(models_all,mode='sum'))  # 将所有的model合并 merge
# 	model.add(Dropout(0.5))
# 	model.add(Dense(nb_classes))   # 输出层 64->7
# 	model.add(Activation('softmax'))

# 	model.compile(loss='categorical_crossentropy',optimizer='adam')  #  配置网络 adam
#     # 可以写个for循环 记录做大的acc 或 G-mean

# 	hist=model.fit(X_train,Y_train[0],batch_size=20,nb_epoch=nb_epoch,verbose=1,shuffle=True,
# 		          show_accuracy=True,validation_data=(X_test,Y_test[0]))   # validation_data=(X_test,Y_test)  训练网络

# 	print(hist.history)   # 输出历史记录
# 	best_acc=max(hist.history['val_acc'])  # 输出最好的验证正确率
# 	print('the best epoch:%d,and the acc:%.5f' %(hist.history['val_acc'].index(best_acc)+1,best_acc))

# 	f=open('acc_record.txt','a')         # 将结果保存到文件
# 	f.write('------------%d epoch---------------\n' %(nb_epoch))
# 	f.write('val_acc: '+str(hist.history['val_acc'])+'\n')
# 	f.write('the best epoch:%d,and the acc:%.5f\n' %(hist.history['val_acc'].index(best_acc)+1,best_acc))
# 	f.write('-----------------------------------\n')
# 	f.close()

# 	pred_label=model.predict_classes(X_test,batch_size=20,verbose=1)  # 对测试样本进行类别预测
# 	write2File(pred_label,test_label[0])
# 	return pred_label,test_label[0]

nb_epoch=50

def Multi_channel(models_all,X_train,X_test,Y_train,Y_test,test_label,X_val,Y_val):  # multi-channel 的神经网络
	print('开始Merge所有的通道...')
	model=Sequential()
	model.add(Merge(models_all,mode='sum'))  # 将所有的model合并 merge
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))   # 输出层 64->7
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #  配置网络 adam rmsprop 0.49206
	# 可以写个for循环 记录做大的acc 或 G-mean
	# X_val=[]
	# X_train2=[]
	# Y_val=[]
	# Y_train2=[]
	# for j in range(5):
	# 	x_val=[]
	# 	y_val=[]
	# 	x_tra=[]
	# 	y_tra=[]
	# 	for i in range(7):
	# 		x_val.append(X_train[j][288*i+228:288*i+288])
	# 		x_tra.append(X_train[j][288*i:288*i+228])
	# 		y_val.append(Y_train[j][288*i+228:288*i+288])
	# 		y_tra.append(Y_train[j][288*i:288*i+228])
	# 	x_val=np.array(X_val)
	# 	# x_val=np_utils.to_categorical(x_val,nb_classes)
	# 	# y_val=np.array(Y_val)
	# 	y_val=np_utils.to_categorical(y_val,nb_classes)
	# 	x_tra=np.array(x_tra)
	# 	# y_tra=np.array(y_tra)
	# 	y_tra=np_utils.to_categorical(y_tra,nb_classes)
	# 	X_val.append(x_val)
	# 	Y_val.append(y_val)
	# 	X_train2.append(x_tra)
	# 	Y_train2.append(y_tra)
	best_acc=0
	best_pred_label=[]
	best_epoch=0
	a0=0
	b0=0
	val_acc_list=[]
	test_acc_list=[]
	for i in range(nb_epoch):
		print('%d epoch...' %(i+1))
		print('Now:best_acc:%.5f \t best_epoch:%d' %(best_acc,best_epoch))
		hist=model.fit(X_train,Y_train[0], batch_size=32, nb_epoch=1, verbose=1,   #20 10
	                shuffle=True,validation_data=(X_val,Y_val[0]))  # (X_test,Y_test[0])
		acc=max(hist.history['val_acc'])
		val_acc_list.append(str(acc))
		# acc=model.evaluate(X_test,Y_test[0],batch_size=32,show_accuracy=True,verbose=1)[1]
		p_label=model.predict_classes(X_test,batch_size=32,verbose=1)
		# acc=np_utils.accuracy(p_label,Y_test)  # 得到正确率
		print p_label
		a1=np_utils.accuracy(p_label,test_label[0])
		test_acc_list.append(str(a1))
		print 'Now epoch test acc:%.5f' %(a1)
		if a1>a0:
			print 'a1 better:%.5f' %(a1)
			a0=a1
			b0=(i+1)
		# print Y_test[0]
		# print test_label[0]
		if acc>best_acc:
			print('出现更好的acc,正在更新acc和epoch...')
			best_acc=acc
			best_pred_label=p_label
			best_epoch=(i+1)
	test_acc=np_utils.accuracy(best_pred_label,test_label[0])  # 得到正确率
	print('the best val epoch:%d,and the acc:%.5f,while best test acc epoch:%d,%.5f' %(best_epoch,test_acc,b0,a0))
	# hist=model.fit(X_train,Y_train[0],batch_size=20,nb_epoch=nb_epoch,verbose=1,shuffle=True,
	# 	          show_accuracy=True,validation_data=(X_test,Y_test[0]))   # validation_data=(X_test,Y_test)  训练网络

	# print(hist.history)   # 输出历史记录
	# best_acc=max(hist.history['val_acc'])  # 输出最好的验证正确率
	# print('the best epoch:%d,and the acc:%.5f' %(hist.history['val_acc'].index(best_acc)+1,best_acc))

	# f=open('acc_record.txt','a')         # 将结果保存到文件
	# f.write('------------%d epoch---------------\n' %(nb_epoch))
	# f.write('val_acc: '+str(hist.history['val_acc'])+'\n')
	# f.write('the best epoch:%d,and the acc:%.5f\n' %(hist.history['val_acc'].index(best_acc)+1,best_acc))
	# f.write('-----------------------------------\n')
	# f.close()

	# pred_label=model.predict_classes(X_test,batch_size=20,verbose=1)  # 对测试样本进行类别预测
	write2File(best_pred_label,test_label[0])
	f=open('acc_compare.txt','w')
	f.write(' '.join(list(val_acc_list)))
	f.write('\n')
	f.write(' '.join(list(test_acc_list)))
	f.write('\n')
	f.close()
	return best_pred_label,test_label[0],best_epoch

def write2File(pred_label,real_label):  # 将预测结果写入文件
	f=open('multi_result.txt','w')
	for i in range(len(pred_label)):
		f.write('%d\t%d\n' %(real_label[i],pred_label[i]))
	f.close()

def createResult(pred_label,real_label,best_epoch):  # 计算prf值 写入文件
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
    f=open('multi_prf.txt','a')
    f.write('------------------epoch:%d---seed:%d----time:%s------------------------\n' %(nb_epoch,seed,nowTime))
    for i in range(7):
    	print (i)
    	precision[i]+=(tp[i]/(tp[i]+fp[i]))  # 精确率
    	recall[i]+=(tp[i]/p[i])             # 召回率
    	dot*=recall[i]
    	F1[i]+=((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    	f.write('label:%d \t precision:%.5f \t recall:%.5f \t F1:%.5f\n' %(i,precision[i],recall[i],F1[i]))
    G_meam=dot**(1.0/7.0)
    f.write('---------------acc:%.5f---G_mean:%.5f---best_epoch:%d------------------\n\n' %(acc,G_meam,best_epoch))
    f.close()

if __name__ == '__main__':
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
	# X_train_1,X_test_1,Y_train_1,Y_test_1,test_label_1=load_data(r'Model/model.m',em0,em1,em2,em3,em4,em5,em6)
	# X_train_2,X_test_2,Y_train_2,Y_test_2,test_label_2=load_data(r'Model/model.m',em0,em1,em2,em3,em4,em5,em6)
	# X_train_3,X_test_3,Y_train_3,Y_test_3,test_label_3=load_data(r'Model/model.m',em0,em1,em2,em3,em4,em5,em6)
	# X_train_4,X_test_4,Y_train_4,Y_test_4,test_label_4=load_data(r'Model/model.m',em0,em1,em2,em3,em4,em5,em6)
	# X_train_5,X_test_5,Y_train_5,Y_test_5,test_label_5=load_data(r'Model/model.m',em0,em1,em2,em3,em4,em5,em6)
	# # X_train,X_test,Y_train,Y_test,test_label=load_data(r'Model/model.m',em0,em1,em2,em3,em4,em5,em6)
	# # LSTM_model(X_train,X_test,Y_train,Y_test,test_label)
	# model_1=LSTM_model(X_train_1,X_test_1,Y_train_1,Y_test_1,test_label_1)
	# model_2=LSTM_model(X_train_1,X_test_1,Y_train_1,Y_test_1,test_label_1)
	# model_3=LSTM_model(X_train_1,X_test_1,Y_train_1,Y_test_1,test_label_1)
	# model_4=LSTM_model(X_train_1,X_test_1,Y_train_1,Y_test_1,test_label_1)
	# model_5=LSTM_model(X_train_1,X_test_1,Y_train_1,Y_test_1,test_label_1)

	models_all,X_train,X_test,Y_train,Y_test,test_label,X_val,Y_val=getAllModels(my_model,em0,em1,em2,em3,em4,em5,em6)
	pred_label,real_label,best_epoch=Multi_channel(models_all,X_train,X_test,Y_train,Y_test,test_label,X_val,Y_val)
	# print(pred_label)
	# print(real_label)
	createResult(pred_label,real_label,best_epoch)


