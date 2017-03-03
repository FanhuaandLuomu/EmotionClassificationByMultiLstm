#coding:utf-8

class Document:
	def __init__(self,polarity,words):
		self.polarity=polarity   # 类别
		self.words=words   #词特征
 
def readFromFiles(path,polarity):  #创建文档列表
	lines=open(path,'rb')
	documents=[]
	for line in lines:
		pieces=line.strip().split(' ')
		#pieces=[piece for piece in pieces if piece.isalpha()]
		words={}
		for piece in pieces:
			word=piece.lower()
			if(word not in words):
				words[word]=1
		documents.append(Document(polarity,words))
	return documents

def createDomain():
	e0=readFromFiles(r'raw_corpus1_split/resort_anger.txt',0)  # anger
	e1=readFromFiles(r'raw_corpus1_split/resort_disgust.txt',1)  # disgust
	e2=readFromFiles(r'raw_corpus1_split/resort_fear.txt',2)  # fear
	e3=readFromFiles(r'raw_corpus1_split/resort_happiness.txt',3)  # happiness
	e4=readFromFiles(r'raw_corpus1_split/resort_like.txt',4)  # like
	e5=readFromFiles(r'raw_corpus1_split/resort_sadness.txt',5)  # sadness
	e6=readFromFiles(r'raw_corpus1_split/resort_surprise.txt',6)  # surprise
	return e0,e1,e2,e3,e4,e5,e6

# neg,pos=createDomain()
# print len(neg[0].words.keys())
# for key in neg[0].words:
# 	print key

# domain=createDomain()
# for item in domain:
# 	print len(item)
