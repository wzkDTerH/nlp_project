import FLAGS
class dataIO:
	def __init__(self,filename,datatype):
		self.dataType=datatype
		if(datatype!='train' and datatype!='test'):
			raise ValueError("datatype must be train or test!")
		self.filename=filename
		self.Qnum=20
		self.LoadData()
	def LoadData(self):
		print 'Loading %s data %s'%(self.dataType,self.filename)
		fp=open(self.filename,'r')
		self.dataQuery=[]
		self.dataSupport=[]
		self.dataOption=[]
		self.dataAnswer=[]
		self.dataNum=0
		self.maxSupportLen=0
		self.maxQueryLen=0
		self.optionLen=0
		while(True):
			self.dataNum=self.dataNum+1
			if(self.dataNum%500==0):
				print '\rSample Count:%d'%(self.dataNum),;
			self.dataSupport.append([])
			for i in range(self.Qnum):
				line=fp.readline()[:-1].split(' ')
				self.dataSupport[-1].extend(line[1:])
			line=fp.readline()[:-1].split('\t')
	 		self.dataQuery.append(line[0].split(' ')[1:])
			self.dataOption.append(line[-1].split('|'))
			if(self.dataType=='train'):
				self.dataAnswer.append(line[1])

			self.maxSupportLen=max(self.maxSupportLen,len(self.dataSupport[-1]))
			self.maxQueryLen=max(self.maxQueryLen,len(self.dataQuery[-1]))
			self.optionLen=len(self.dataOption[-1])
			line=fp.readline()
			if(type(line)==type(None) or line==''):
				break
		print '\rSample Count:%d,maxSupportLen=%d,maxQueryLen=%d'%(self.dataNum,self.maxSupportLen,self.maxQueryLen);
		fp.close()
		print 'Load %s data %s done'%(self.dataType,self.filename)
if(__name__=='__main__'):
	pass