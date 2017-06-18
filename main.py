import FLAGS
import rnn_model

if(__name__=='__main__'):
	model=rnn_model.rnn_model()
	if(FLAGS.FLAGS.train):
		model.Train()
	if(FLAGS.FLAGS.answer):
		model.Run()
