import random
import numpy as np
import FLAGS

class wordEmbedding:
	def __init__(self,filedir=FLAGS.FLAGS.vectorName):
		self.filedir=filedir
		self.LoadVector()
	def LoadVector(self):
		print 'Loading Vector file...',;
		self.vectorFile=open(self.filedir,'r')
		print '\rLoading Vector file into Dict';
		self.vectorDict={}
		self.vectorLen=-1
		self.wordNum=0
		self.maxK,self.minK=0,0
		for line in self.vectorFile.readlines():
			self.wordNum=self.wordNum+1
			if(self.wordNum%500==0):
				print '\rWord Count:%d'%(self.wordNum),;
			line=line.split(' ')
			word=line[0]
			self.vectorDict[word]=[]
			for number in line[1:]:
				num=float(number)
				self.vectorDict[word].append(num)
				self.maxK,self.minK=max(self.maxK,num),min(self.minK,num)
			self.vectorDict[word]=np.array(self.vectorDict[word])
			if(self.vectorLen==-1):
				self.vectorLen=len(self.vectorDict[word])
		print '\rWord Count:%d ,VectorLen=%d'%(self.wordNum,self.vectorLen);
		FLAGS.FLAGS.vectorLen=self.vectorLen
		self.vectorFile.close()
		print 'Load Vector file done             '

	def __call__(self,word):
		if(not self.vectorDict.has_key(word)):
			self.vectorDict[word]=np.array([round(random.uniform(self.minK,self.maxK),6)
			                                for i in range(self.vectorLen)])
		return self.vectorDict[word]
	def __del__(self):
		pass

if(__name__=='__main__'):
	pass