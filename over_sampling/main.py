#coding:utf-8
from __future__ import division
from document import createDomain
import maxent
import random
import time
ISOTIMEFORMAT='%Y-%m-%d %X'
# seed=2222
# random.seed(seed)

def run(seed):
	domain=createDomain()
	l0=len(domain[0])
	l1=len(domain[1])
	l2=len(domain[2])
	l3=len(domain[3])
	l4=len(domain[4])
	l5=len(domain[5])
	l6=len(domain[6])
	print('len(e0):'+str(len(domain[0])))
	print('len(e1):'+str(len(domain[1])))
	print('len(e2):'+str(len(domain[2])))
	print('len(e3):'+str(len(domain[3])))   #  472  470 0.8 376
	print('len(e4):'+str(len(domain[4])))
	print('len(e5):'+str(len(domain[5])))
	print('len(e6):'+str(len(domain[6])))
	
	# docs=domain[0]+domain[1]+domain[2]+domain[3]+domain[4]+domain[5]+domain[6]
	# trains=domain[0][int(l0*0.2):]+domain[1][int(l1*0.2):]+domain[2][int(l2*0.2):]+\
	# 		domain[3][int(l3*0.2):]+domain[4][int(l4*0.2):]+domain[5][int(l5*0.2):]+domain[6][int(l6*0.2):]
	tests=domain[0][:72]+domain[1][:72]+domain[2][:72]+domain[3][:72]+domain[4][:72]\
			+domain[5][:72]+domain[6][:72]	
	domain_2=[]   # re-sampling
	for j in range(6):
		for i in range(2204-len(domain[j])):
			domain[j].append(random.choice(domain[j][72:]))
	# 测试样本  固定每类别的前90 *7
	trains=domain[0][72:]+domain[1][72:]+domain[2][72:]+domain[3][72:]+domain[4][72:]\
			+domain[5][72:]+domain[6][72:]

	# for item in domain:
	# 	random.shuffle(item)
	# 训练样本  随机采样 360 *7
	# trains=random.sample(domain[0][72:],288)+random.sample(domain[1][72:],288)+random.sample(domain[2][72:],288)\
	# 		+random.sample(domain[3][72:],288)+random.sample(domain[4][72:],288)\
	# 	     +random.sample(domain[5][72:],288)+random.sample(domain[6][72:],288)

	# random.shuffle(trains)
	# tests=domain[0][:int(l0*0.2)]+domain[1][:int(l1*0.2)]+domain[2][:int(l2*0.2)]+domain[3][:int(l3*0.2)]+\
	#  	   domain[4][:int(l4*0.2)]+domain[5][:int(l5*0.2)]+domain[6][:int(l6*0.2)]

	#random.shuffle(tests)
	#random.shuffle(docs)
	# trains=docs[:int(len(docs)*0.8)]
	# tests=docs[int(len(docs)*0.8):]
	print('len(trains):'+str(len(trains)))
	print('len(tests):'+str(len(tests)))

	# lexcion=maxent.get_lexcion(trains)
	# print('len(lexcion):'+str(len(lexcion)))  

	maxent.me_classify(trains,tests)
	#maxent.createResult(tests,'result.txt')
	# acc=maxent.createResult2('result.txt')
	acc,G_mean=maxent.createPRF('result.txt',seed)
	return acc,G_mean
# run()
if __name__ == '__main__':
	count=10
	acc_sum=0
	g_sum=0
	acc_list=[]
	seed=[1,1111,2222,3333,4444,5555,6666,7777,8888,9999]
	# seed=[2222]
	for i in range(len(seed)):
		random.seed(seed[i])
		acc,G_mean=run(seed[i])
		acc_list.append(acc)
		acc_sum+=acc
		g_sum+=G_mean
	average_acc=acc_sum/count
	average_gmean=g_sum/count
	print acc_list
	print 'average_acc:%.5f' %(average_acc)
	print 'average_gmean:%.5f' %(average_gmean)

	# f=open('acc_record','a')
	# f.write('---------------count:%d-----------------\n' %count)
	# f.write('acc:'+str(acc_list)+'\n')
	# f.write('average:'+str(average)+'\n')
	# f.write('----------------------------------------\n')
	# f.close()
	# print 'average:%.3f' %(average)