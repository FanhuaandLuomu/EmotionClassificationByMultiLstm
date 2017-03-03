# coding:utf-8
from __future__ import division
from document import createDomain
import maxent
import random
# seed=4444
# random.seed(seed)
import time
ISOTIMEFORMAT='%Y-%m-%d %X'

def getTrains(domain):
	trains=random.sample(domain[0][72:],288)+random.sample(domain[1][72:],288)+random.sample(domain[2][72:],288)\
			+random.sample(domain[3][72:],288)+random.sample(domain[4][72:],288)\
		     +random.sample(domain[5][72:],288)+random.sample(domain[6][72:],288)
	return trains

def run(trains,tests,filename,seed):
	maxent.me_classify(trains,tests,filename)
	pred_prob,pred_label,real_label=maxent.getPredProb(filename)
	acc,gmean=maxent.createPRF(pred_label,real_label,seed,'per_crf.txt')
	return pred_prob,real_label
	
def main(seed):
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

	tests=domain[0][:72]+domain[1][:72]+domain[2][:72]+domain[3][:72]+domain[4][:72]\
			+domain[5][:72]+domain[6][:72]
	domain_2=[]   # re-sampling
	for i in range(360-len(domain[2][72:])): 
		domain[2].append(random.choice(domain[2][72:]))
	# domain[2]=domain_2
	
	# trainList=[]  # 训练样本列表
	pred_prob_list=[]
	co_num=5
	for i in range(co_num):
		# trainList.append(getTrains(domain))
		trains=getTrains(domain)
		pred_prob,real_label=run(trains,tests,'result_%d.txt' %(i+1),seed)
		pred_prob_list.append(pred_prob)
	co_pred_prob=[]
	print len(pred_prob_list)

	for i in range(len(tests)):
		p=[]
		# p0=pred_prob_list[0][i][0]+pred_prob_list[1][i][0]+pred_prob_list[2][i][0]+pred_prob_list[3][i][0]+\
		# 	pred_prob_list[4][i][0]
		p0=sum([item[i][0] for item in pred_prob_list])
		p1=sum([item[i][1] for item in pred_prob_list])
		p2=sum([item[i][2] for item in pred_prob_list])
		p3=sum([item[i][3] for item in pred_prob_list])
		p4=sum([item[i][4] for item in pred_prob_list])
		p5=sum([item[i][5] for item in pred_prob_list])
		p6=sum([item[i][6] for item in pred_prob_list])
		# p1=pred_prob_list[0][i][1]+pred_prob_list[1][i][1]+pred_prob_list[2][i][1]+pred_prob_list[3][i][1]+\
		# 	pred_prob_list[4][i][1]
		# p2=pred_prob_list[0][i][2]+pred_prob_list[1][i][2]+pred_prob_list[2][i][2]+pred_prob_list[3][i][2]+\
		# 	pred_prob_list[4][i][2]
		# p3=pred_prob_list[0][i][3]+pred_prob_list[1][i][3]+pred_prob_list[2][i][3]+pred_prob_list[3][i][3]+\
		# 	pred_prob_list[4][i][3]
		# p4=pred_prob_list[0][i][4]+pred_prob_list[1][i][4]+pred_prob_list[2][i][4]+pred_prob_list[3][i][4]+\
		# 	pred_prob_list[4][i][4]
		# p5=pred_prob_list[0][i][5]+pred_prob_list[1][i][5]+pred_prob_list[2][i][5]+pred_prob_list[3][i][5]+\
		# 	pred_prob_list[4][i][5]
		# p6=pred_prob_list[0][i][6]+pred_prob_list[1][i][6]+pred_prob_list[2][i][6]+pred_prob_list[3][i][6]+\
		# 	pred_prob_list[4][i][6]
		p=[p0,p1,p2,p3,p4,p5,p6]
		co_pred_prob.append(p)
	# print co_pred_prob[0]
	print len(co_pred_prob)
	co_max_prob_label=[]
	for item in co_pred_prob:
		m=max(item)
		co_max_prob_label.append(item.index(m))
		# co_max_prob.append(max(item))
	print co_max_prob_label
	acc,g_mean=maxent.createPRF(co_max_prob_label,real_label,seed,'co_resort_maxent_prf.txt')
	print acc,g_mean
	return acc,g_mean

if __name__ == '__main__':
	seed=[1,1111,2222,3333,4444,5555,6666,7777,8888,9999]
	# seed=[9999]
	accList=[]
	sum_acc=0
	sum_gmean=0
	for item in seed:
		random.seed(item)
		print('seed:',item)
		acc,G_mean=main(item)
		sum_acc+=acc
		sum_gmean+=G_mean
		accList.append(acc)
	for i in range(len(seed)):
		print('%d:%.5f' %(seed[i],accList[i]))
	ave_acc=sum_acc/len(seed)
	ave_gmean=sum_gmean/len(seed)
	print 'ave_acc:%.5f' %(ave_acc)
	print 'ave_gmean:%.5f' %(ave_gmean)


