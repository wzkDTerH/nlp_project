import dataIO
import wordEmbedding
import numpy as np
from random import sample
import FLAGS

class dataSet:

	wordVector=wordEmbedding.wordEmbedding(FLAGS.FLAGS.vectorName)

	def __init__(self,filename,dataType):
		self.dataType=dataType
		if(dataType!='train' and dataType!='test'):
			raise ValueError("datatype must be train or test!")
		self.filename=filename
		self.data=dataIO.dataIO(self.filename,dataType)
		self.optionLen=self.data.optionLen
		if(FLAGS.FLAGS.maxSupportLen<self.data.maxSupportLen):
			FLAGS.FLAGS.maxSupportLen=self.data.maxSupportLen
		self.maxSupportLen=FLAGS.FLAGS.maxSupportLen
		if(FLAGS.FLAGS.maxQueryLen<self.data.maxQueryLen):
			FLAGS.FLAGS.maxQueryLen=self.data.maxQueryLen
		self.maxQueryLen=FLAGS.FLAGS.maxQueryLen
		self.batchStart=0
		self.vectorLen=dataSet.wordVector.vectorLen

	def End(self):
		return self.batchStart>=self.data.dataNum

	def nextBatchs(self,batchSize=FLAGS.FLAGS.batchSize,random=False):
		retSupport=np.zeros([batchSize,self.maxSupportLen,self.vectorLen])
		retSupportLen=np.zeros([batchSize])
		retQuery=np.zeros([batchSize,self.maxQueryLen,self.vectorLen])
		retQueryLen=np.zeros([batchSize])
		retOption=[]
		retOptionSupportMask=np.zeros([batchSize,self.optionLen,self.maxSupportLen])
		retAnswer=np.zeros([batchSize,self.optionLen])
		if(random==True):
			samples=sample(range(self.data.dataNum),batchSize)
		else:
			samples=range(self.batchStart,min(self.batchStart+batchSize,self.data.dataNum))
			self.batchStart=self.batchStart+batchSize
		for i in range(len(samples)):
			x=samples[i]
			retSupportLen[i]=len(self.data.dataSupport[x])
			for j in range(int(retSupportLen[i])):
				retSupport[i,j]=dataSet.wordVector(self.data.dataSupport[x][j])
				for k in range(self.data.optionLen):
					if(self.data.dataOption[x][k]==self.data.dataSupport[x][j]):
						retOptionSupportMask[i][k][j]=1
			retQueryLen[i]=len(self.data.dataQuery[x])
			for j in range(int(retQueryLen[i])):
				retQuery[i,j]=dataSet.wordVector(self.data.dataQuery[x][j])
			retOption.append([])
			for j in range(self.data.optionLen):
				retOption[-1].append(self.data.dataOption[x][j])
			if(self.dataType=='train'):
				for j in range(self.data.optionLen):
					if(self.data.dataAnswer[x]==self.data.dataOption[x][j]):
						retAnswer[i][j]=1
		return retSupport,retSupportLen,retQuery,retQueryLen,retOption,retOptionSupportMask,retAnswer
