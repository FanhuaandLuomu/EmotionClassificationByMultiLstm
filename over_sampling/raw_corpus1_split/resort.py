# coding:utf-8
import threading
import os
import random
# random.seed(1111) # 多线程时没什么用了

def readFromFileAndWrite(filename1,filename2):
	lines=[]
	f=open(filename1,'r')
	for line in f:
		lines.append(line.strip())
	write2File(filename2,lines)

def write2File(filename,lines):
	f=open(filename,'w')
	random.shuffle(lines)
	text='\n'.join(lines)
	f.write(text)
	f.close()

if __name__ == '__main__':
	filenames=os.listdir('./')
	filenames.remove('resort.py')
	print filenames
	threads=[]
	for filename in filenames:
		t=threading.Thread(target=readFromFileAndWrite,args=(filename,'resort_'+filename))
		threads.append(t)
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	print 'all file resort success...END'
