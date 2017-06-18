import tensorflow as tf
import dataSet
import FLAGS
import os

class rnn_model:
	def __init__(self):
		self.sess = tf.Session()
		self.builded=False
	def __del__(self):
		self.sess.close()
	def BuildModel(self):
		if(self.builded): return
		self.builded=True
		batchSize=FLAGS.FLAGS.batchSize
		maxSupportLen=FLAGS.FLAGS.maxSupportLen
		maxQueryLen=FLAGS.FLAGS.maxQueryLen
		vectorLen=FLAGS.FLAGS.vectorLen
		optionLen=FLAGS.FLAGS.optionLen
		stateSize=FLAGS.FLAGS.stateSize
		EPSILON=FLAGS.FLAGS.EPSILON
		with tf.name_scope('Input'):
			self.support = tf.placeholder(tf.float32, [None, maxSupportLen,vectorLen],name='support')
			self.query=tf.placeholder(tf.float32,[None,maxQueryLen,vectorLen],name='query')
			self.options=tf.placeholder(tf.float32,[None,optionLen,vectorLen],name='options')
			self.supportLen=tf.placeholder(tf.int32,[None],name='supportLen')
			self.queryLen=tf.placeholder(tf.int32,[None],name='queryLen')
			self.optionSupportMask=tf.placeholder(tf.float32,[None,optionLen,maxSupportLen],name='optionSupportMask')
			self.y_std_onthot=tf.placeholder(tf.float32,[None,optionLen],name='y_std_onthot')
		biDirectRnn=tf.nn.bidirectional_dynamic_rnn
		with tf.variable_scope('support', initializer=tf.orthogonal_initializer()):
			self.support_rnn_fw=tf.contrib.rnn.GRUCell(num_units=stateSize)
			self.support_rnn_bw=tf.contrib.rnn.GRUCell(num_units=stateSize)
			(self.support_outs,states)=biDirectRnn(cell_fw=self.support_rnn_fw,
			                                       cell_bw=self.support_rnn_bw,
			                                       swap_memory=True,
			                                       dtype="float32",
			                                       inputs=self.support,
			                                       sequence_length=self.supportLen)
			self.support_outputs=tf.concat(self.support_outs, -1)
		with tf.variable_scope('query', initializer=tf.orthogonal_initializer()):
			self.query_rnn_fw=tf.contrib.rnn.GRUCell(num_units=stateSize)
			self.query_rnn_bw=tf.contrib.rnn.GRUCell(num_units=stateSize)
			(outs,self.query_states)=biDirectRnn(cell_fw=self.query_rnn_fw,
			                                     cell_bw=self.query_rnn_bw,
			                                     swap_memory=True,
			                                     dtype="float32",
			                                     inputs=self.query,
			                                     sequence_length=self.queryLen)
			self.query_states_fw,self.query_states_bw=self.query_states
			self.query_outputs=tf.concat([self.query_states_fw,
			                              self.query_states_bw],axis=-1)

		with tf.name_scope('Attention'):
			def attentionDot(support_bf,query_bf):
				ret = tf.matmul(support_bf,tf.expand_dims(query_bf, -1))
				return tf.reshape(ret, [-1, maxSupportLen])
			self.attention_dot = attentionDot(self.support_outputs, self.query_outputs)
			with tf.name_scope('Support_Mask'):
				self.support_mask = tf.sequence_mask(self.supportLen, maxSupportLen, dtype=tf.float32)
				self.attention_dot_mask=tf.multiply(self.attention_dot,self.support_mask)
			self.attention_weight=tf.nn.softmax(self.attention_dot_mask,
			                                    name='attention_softmax_weight')
		with tf.name_scope('OptionProb'):
			self.word_attention=tf.reshape(tf.matmul(self.optionSupportMask,tf.expand_dims(self.attention_weight,-1)),
			                               [-1,optionLen])
			self.option_output = self.word_attention / tf.reduce_sum(self.word_attention, axis=-1, keep_dims=True)
		with tf.name_scope('Output'):
			self.epsilon = tf.convert_to_tensor(EPSILON, self.option_output.dtype.base_dtype, name="epsilon")
			self.output = tf.clip_by_value(self.option_output, self.epsilon, 1. - self.epsilon)
		with tf.name_scope('loss'):
			self.loss =-tf.reduce_mean(tf.reduce_sum(self.y_std_onthot * tf.log(self.output), axis=-1))
			tf.summary.scalar('loss',self.loss)
		with tf.name_scope('correct_prediction'):
			self.prediction=tf.argmax(self.output, 1)
			self.correct_prediction = tf.reduce_mean(tf.sign(tf.cast(tf.equal(self.prediction,
			                                                                 tf.argmax(self.y_std_onthot, 1)), "float")))
			tf.summary.scalar('correct_prediction_rate',self.correct_prediction)
		with tf.name_scope('train_step'):
			self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
		with tf.name_scope('saver'):
			self.saver = tf.train.Saver(max_to_keep=20)

	def saveModel(self, ac, step):
		path=self.saver.save(self.sess,
		                     os.path.join(FLAGS.FLAGS.modelPath,"rnnmodel_%0.3f.model"%(ac)),
		                     global_step=step)
		print "Save  model to ",path

	def loadModel(self):
		ckpt = tf.train.get_checkpoint_state(FLAGS.FLAGS.modelPath)
		if ckpt is not None:
			path=tf.train.latest_checkpoint(FLAGS.FLAGS.modelPath)
			print "Load  model from ",path
			self.saver.restore(self.sess, path)
			return int(path.split('-')[-1])+1
		else:
			print "No model to Load"
			return 0

	def Answer(self,T):
		print "Answering",FLAGS.FLAGS.reasultDir+'reasult_%d.txt'%(T)
		fp=open(FLAGS.FLAGS.reasultDir+'reasult_%d.txt'%(T),'w')
		self.testData.batchStart=0
		while(not self.testData.End()):
			batch=self.testData.nextBatchs(random=False)
			support,supportLen,query,queryLen,option,optionSupportMask,answer=batch
			feed_dict={self.support: support,
			           self.supportLen:supportLen,
			           self.query:query,
			           self.queryLen:queryLen,
			           self.optionSupportMask:optionSupportMask,
			           self.y_std_onthot:answer}
			prediction=self.sess.run(self.prediction,feed_dict=feed_dict)
			for i in range(len(option)):
				fp.write(option[i][int(prediction[i])]+'\n')
		fp.close()

	def DevAcc(self,T):
		self.devData.batchStart=0
		total=0
		acnum=0.0
		while(not self.devData.End()):
			batch=self.devData.nextBatchs(random=False)
			support,supportLen,query,queryLen,option,optionSupportMask,answer=batch
			feed_dict={self.support: support,
			           self.supportLen:supportLen,
			           self.query:query,
			           self.queryLen:queryLen,
			           self.optionSupportMask:optionSupportMask,
			           self.y_std_onthot:answer}
			acrate=self.sess.run(self.correct_prediction,feed_dict=feed_dict)
			total=total+len(option)
			acnum=acnum+len(option)*acrate
		res='Dev AC:total %d ,acnum %d acrate %f'%(int(total),int(acnum),acnum/total)
		print res
		fp=open(FLAGS.FLAGS.reasultDir+'devAc_%d.txt'%(T),'w')
		fp.write(res)
		fp.close()

	def Run(self):
		import sys
		self.devData=dataSet.dataSet(FLAGS.FLAGS.devDataName,'train')
		self.sess.run(tf.global_variables_initializer())
#		self.BuildModel()
		ckpt = tf.train.get_checkpoint_state(FLAGS.FLAGS.modelPath)
		if ckpt is not None:
			path=tf.train.latest_checkpoint(FLAGS.FLAGS.modelPath)
			print "Load  model from ",path
			path='./model/rnnmodel_0.594.model-9700'
			self.saver = tf.train.import_meta_graph(path+'.meta')
			self.saver.restore(self.sess, path)
		else:
			print "No model to Load"
			exit(0)
		graph = tf.get_default_graph()
		self.support=graph.get_tensor_by_name("Input/support:0")
		self.supportLen=graph.get_tensor_by_name("Input/supportLen:0")
		self.query=graph.get_tensor_by_name("Input/query:0")
		self.queryLen=graph.get_tensor_by_name("Input/queryLen:0")
		self.optionSupportMask=graph.get_tensor_by_name("Input/optionSupportMask:0")
		self.y_std_onthot=graph.get_tensor_by_name("Input/y_std_onthot:0")
		self.correct_prediction=graph.get_tensor_by_name("correct_prediction/Mean:0")
		total=0
		acnum=0.0
		T=0
		while(not self.devData.End()):
			batch=self.devData.nextBatchs(random=False)
			support,supportLen,query,queryLen,option,optionSupportMask,answer=batch
			feed_dict={self.support: support,
			           self.supportLen:supportLen,
			           self.query:query,
			           self.queryLen:queryLen,
			           self.optionSupportMask:optionSupportMask,
			           self.y_std_onthot:answer}
			acrate=self.sess.run(self.correct_prediction,feed_dict=feed_dict)
			total=total+len(option)
			acnum=acnum+len(option)*acrate
			print 'T',T,"total",total,"acnum",acnum,"acrate",acnum/total
			T=T+1

	def Train(self):
		import sys
		self.trainData=dataSet.dataSet(FLAGS.FLAGS.trainDataName,'train')
		self.devData=dataSet.dataSet(FLAGS.FLAGS.devDataName,'train')
		self.testData=dataSet.dataSet(FLAGS.FLAGS.testDataName,'test')
		self.BuildModel()
		self.sess.run(tf.global_variables_initializer())
		T0=self.loadModel()
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("./graph_train/",self.sess.graph)
		def GetFeedDict(batch):
			support,supportLen,query,queryLen,option,optionSupportMask,answer=batch
			return {self.support: support,
			        self.supportLen:supportLen,
			        self.query:query,
			        self.queryLen:queryLen,
			        self.optionSupportMask:optionSupportMask,
			        self.y_std_onthot:answer}
		T=T0
		acnum=0
		for T in range(T0,T0+FLAGS.FLAGS.trainTrun):
			print 'T:%04d|'%(T),
			sys.stdout.flush()
			if(T%10==0):
				batch=self.devData.nextBatchs(random=True)
				feed_dict=GetFeedDict(batch)
				summary,acnum,loss=self.sess.run([self.merged,self.correct_prediction,self.loss],
				                                 feed_dict=feed_dict)
				print 'loss',loss,'acrate',acnum
				self.writer.add_summary(summary,T)
			batch=self.trainData.nextBatchs(random=True)
			feed_dict=GetFeedDict(batch)
			self.sess.run(self.train_step, feed_dict=feed_dict)
			if(T%100==0):
				self.saveModel(ac=acnum,step=T)
			if(T%500==0):
				self.Answer(T)
				self.DevAcc(T)
		self.saveModel(ac=acnum,step=T)