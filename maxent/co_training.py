# coding:utf-8
from __future__ import division
from document import createDomain
import maxent
import random
random.seed(9999)
import time
ISOTIMEFORMAT='%Y-%m-%d %X'

def getTrains(domain):
	trains=random.sample(domain[0][72:],288)+random.sample(domain[1][72:],288)+random.sample(domain[2][72:],288)\
			+random.sample(domain[3][72:],288)+random.sample(domain[4][72:],288)\
		     +random.sample(domain[5][72:],288)+random.sample(domain[6][72:],288)
	return trains

def run(trains,tests):
	maxent.me_classify(trains,tests)
	
	
def main():
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

	domain_2=[]   # re-sampling
	for i in range(360):
		domain_2.append(random.choice(domain[2]))
	domain[2]=domain_2
	tests=domain[0][:72]+domain[1][:72]+domain[2][:72]+domain[3][:72]+domain[4][:72]\
			+domain[5][:72]+domain[6][:72]
	trainList=[]  # 训练样本列表
	for i in range(5):
		trainList.append(getTrains(domain))


