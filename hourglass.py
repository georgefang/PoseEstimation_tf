# -*- coding: utf-8 -*-
"""
Human 2D Pose Estimation

Project by Xiao-Zhi Fang
AvatarWorks Lab
Huanshi Ltd.

@author: Xiao-Zhi Fang
@mail : george.fang@avatarworks.com

Abstract:
	This python code creates a Stacked Hourglass Model
	(Credits : A.Newell et al.)
	(Paper : https://arxiv.org/abs/1603.06937)
	
	Code translated from 'anewell' github
	Torch7(LUA) --> TensorFlow(PYTHON)
	(Code : https://github.com/anewell/pose-hg-train)

"""
from utils2d import show_joints
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import cv2

class HourglassModel():
	""" Hourglass Networks
	Generate TensorFlow model to train and predict Human Pose from images
	"""
	def __init__(self, params, dataset=None, training=True, w_summary=True):
		""" Initializer
		Args:
			params: config parameters
			dataset: mpii image dataset
			training: training or inferening
		"""
		self.nStack = params['nstacks']
		self.nFeat = params['nfeats']
		self.nModules = params['nmodules']
		self.outDim = params['num_joints']
		self.batchSize = params['batch_size']
		self.input_res = params['img_size']
		self.output_res = params['hm_size']
		self.training = training
		self.w_summary = w_summary
		self.learning_rate = params['learning_rate']
		self.decay = params['learning_rate_decay']
		self.name = params['name']
		self.mobile = params['mobile']
		self.decay_step = params['decay_step']
		self.nLow = params['nlow']
		self.dataset = dataset
		self.cpu = '/cpu:0'
		self.gpu = '/gpu:0'
		self.logdir_train = params['log_dir_train']
		self.logdir_test = params['log_dir_test']
		self.save_dir = params['saver_directory']
		self.joints = params['joint_list']
		self.w_loss = params['weighted_loss']
		self.multi = 6
		if params.has_key('multi'):
			self.multi = params['multi']
		self.accIdx = np.array([0,1,2,3,4,5,10,11,14,15])
	
	
	def create_model(self):
		""" Create the complete graph
		"""
		startTime = time.time()
		print('CREATE MODEL:')
		with tf.device(self.gpu):
			with tf.name_scope('inputs'):
				# Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
				self.img = tf.placeholder(dtype= tf.float32, shape= (None, self.input_res, self.input_res, 3), name = 'input_img')
				if self.w_loss:
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
				# Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
				self.gtMaps = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, self.output_res, self.output_res, self.outDim))
				# TODO : Implement weighted loss function
				# NOT USABLE AT THE MOMENT
				#weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
			inputTime = time.time()
			print('---Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')
			if self.mobile:
				self.output = self._graph_hourglass_mobile(self.img)
			else :
				self.output = self._graph_hourglass(self.img)
				
			graphTime = time.time()
			print('---Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')
			with tf.name_scope('loss'):
				if self.w_loss:
					self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
				else:
					self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
					# self.loss = tf.reduce_mean(tf.square(self.output - self.gtMaps))
			lossTime = time.time()	
			print('---Loss : Done (' + str(int(abs(graphTime-lossTime))) + ' sec.)')
		with tf.device(self.cpu):
			with tf.name_scope('accuracy'):
				self._accuracy_computation()
			accurTime = time.time()
			print('---Acc : Done (' + str(int(abs(accurTime-lossTime))) + ' sec.)')
			with tf.name_scope('steps'):
				self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
			with tf.name_scope('lr'):
				self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
			lrTime = time.time()
			print('---LR : Done (' + str(int(abs(accurTime-lrTime))) + ' sec.)')
		with tf.device(self.gpu):
			with tf.name_scope('rmsprop'):
				self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.lr)
			optimTime = time.time()
			print('---Optim : Done (' + str(int(abs(optimTime-lrTime))) + ' sec.)')
			with tf.name_scope('minimizer'):
				self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(self.update_ops):
					self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
			minimTime = time.time()
			print('---Minimizer : Done (' + str(int(abs(optimTime-minimTime))) + ' sec.)')
		self.init = tf.global_variables_initializer()
		initTime = time.time()
		print('---Init : Done (' + str(int(abs(initTime-minimTime))) + ' sec.)')
		with tf.device(self.cpu):
			with tf.name_scope('training'):
				tf.summary.scalar('loss', self.loss, collections = ['train'])
				tf.summary.scalar('learning_rate', self.lr, collections = ['train'])
			with tf.name_scope('summary'):
				for i in range(len(self.accIdx)):
					tf.summary.scalar(self.joints[i], self.joint_accur[i], collections = ['train', 'test'])
		self.train_op = tf.summary.merge_all('train')
		self.test_op = tf.summary.merge_all('test')
		self.weight_op = tf.summary.merge_all('weight')
		endTime = time.time()
		print('Model created (' + str(int(abs(endTime-startTime))) + ' sec.)')
		del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime
		
	
	def restore(self, load = None):
		""" Restore a pretrained model
		Args:
			load	: Model to load (None if training from scratch) (see README for further information)
		"""
		# self.Session = tf.Session()
		# print('importing mesh file')
		# self.saver = tf.train.import_meta_graph(load+'.meta')
		# print('loading model')
		# self.saver.restore(self.Session, load)
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_session()
				self._define_saver_summary(summary = False)
				if load is not None:
					print('Loading Trained Model')
					t = time.time()
					# self.saver = tf.train.import_meta_graph(load+'.meta')
					self.saver.restore(self.Session, load)
					print('Model Loaded (', time.time() - t,' sec.)')
				else:
					print('Please give a Model in args (see README for further information)')

	def visualize_valid(self, imgi, hm, set):
		img = np.copy(imgi)
		inp = np.expand_dims(img, axis = 0)
		if set == 'valid':
			out = self.Session.run(self.output, feed_dict = {self.img : inp})
			outhm = out[0,-1]
		elif set == 'train':
			outhm = hm[-1]
		img = cv2.resize(img, (256,256))
		img = img * 255
		img = img.astype(np.uint8)
		outshape = outhm.shape
		assert len(outshape) == 3
		joints = np.zeros((outshape[-1], 2), dtype=np.int64)
		for i in range(0, outshape[-1]):
			resh = np.reshape(outhm[:,:,i], [-1])
			arg = np.argmax(resh)
			#print("hm: {}".format(outhm[:,:,i]))
			joints[i][0] = arg % outshape[1]
			joints[i][1] = arg // outshape[1]
			#print("joint {0}: ({1}, {2})".format(i, joints[i][0], joints[i][1]))
			
		joints = joints * img.shape[0] / outhm.shape[0]
		show_joints(img, joints)

	def _train(self, nEpochs = 10, epochSize = 1000, saveStep = 500, validIter = 10):
		"""
		"""
		with tf.name_scope('Train'):
			startTime = time.time()
			self.resume = {}
			self.resume['accur'] = []
			self.resume['loss'] = []
			self.resume['err'] = []
			for epoch in range(nEpochs):
				ebidx = 0
				randlist = self.dataset.epochsize_cat(epochSize, self.batchSize, sample = 'train')
				epochstartTime = time.time()
				avg_cost = 0.
				cost = 0.
				print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
				# Training Set
				for i in range(epochSize):
					# DISPLAY PROGRESS BAR
					# TODO : Customize Progress Bar
					percent = ((i+1.0)/epochSize) * 100
					num = np.int(20*percent/100)
					tToEpoch = int((time.time() - epochstartTime) * (100 - percent)/(percent))
					sys.stdout.write('\r Train: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
					sys.stdout.flush()
					img_train, gt_train, weight_train = self.dataset._sam_generator(randlist[ebidx:ebidx+self.batchSize], self.batchSize, self.nStack, normalize = True, sample_set = 'train')
					c = 0
					if i % saveStep == 0:
						if self.w_loss:
							_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
						else:
							_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train})
						# Save summary (Loss + Accuracy)
						self.train_summary.add_summary(summary, epoch*epochSize + i)
						self.train_summary.flush()
					else:
						if self.w_loss:
							_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
						else:	
							_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.img : img_train, self.gtMaps: gt_train})
					cost += c
					avg_cost += c/epochSize
					ebidx = ebidx + self.batchSize
				epochfinishTime = time.time()
				#Save Weight (axis = epoch)
				if self.w_loss:
					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
				else :
					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train})
				self.train_summary.add_summary(weight_summary, epoch)
				self.train_summary.flush()
				
				print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime-epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime-epochstartTime)/epochSize))[:4] + ' sec.')
				with tf.name_scope('save'):
					self.saver.save(self.Session, os.path.join(self.save_dir, str(self.name + '_' + str(epoch + 1))))
				self.resume['loss'].append(cost)
				# Validation Set
				ebidx = 0
				accuracy_array = np.array([0.0]*len(self.joint_accur))
				randlist = self.dataset.epochsize_cat(validIter, self.batchSize, sample = 'valid')
				for i in range(validIter):
					img_valid, gt_valid, w_valid = self.dataset._sam_generator(randlist[ebidx:ebidx+self.batchSize], self.batchSize, self.nStack, normalize = True, sample_set = 'valid')
					# img_valid, gt_valid, w_valid = next(self.valid_gen)
					accuracy_pred = self.Session.run(self.joint_accur, feed_dict = {self.img : img_valid, self.gtMaps: gt_valid})
					accuracy_array += np.array(accuracy_pred, dtype = np.float32) / validIter
					ebidx = ebidx + self.batchSize
				print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
				self.resume['accur'].append(accuracy_pred)
				self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
				valid_summary = self.Session.run(self.test_op, feed_dict={self.img : img_valid, self.gtMaps: gt_valid})
				self.test_summary.add_summary(valid_summary, epoch)
				self.test_summary.flush()
				self.visualize_valid(img_valid[0], gt_valid[0], 'valid')
			print('Training Done')
			print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize) )
			print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
			print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
			print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - startTime)))
	

	def record_training(self, record):
		""" Record Training Data and Export them in CSV file
		Args:
			record		: record dictionnary
		"""
		out_file = open(self.name + '_train_record.csv', 'w')
		for line in range(len(record['accur'])):
			out_string = ''
			labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
			for label in labels:
				out_string += str(label) + ', '
			out_string += '\n'
			out_file.write(out_string)
		out_file.close()
		print('Training Record Saved')
			
	def do_train(self, nEpochs = 10, epochSize = 1000, saveStep = 500, dataset = None, load = None):
		""" Do Training
		Args:
			nEpochs		: Number of Epochs to train
			epochSize		: Size of each Epoch
			saveStep		: Step to save 'train' summary (has to be lower than epochSize)
			dataset		: Data Generator (see generator.py)
			load			: Model to load (None if training from scratch) (see README for further information)
		"""
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_weight()
				self._define_saver_summary()
				if load is not None:
					self.saver.restore(self.Session, load)
				self._train(nEpochs, epochSize, saveStep, validIter=10)



	def weighted_bce_loss(self):
		""" Create Weighted Loss Function
		WORK IN PROGRESS
		"""
		self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
		e1 = tf.expand_dims(self.weights,axis = 1, name = 'expdim01')
		e2 = tf.expand_dims(e1,axis = 1, name = 'expdim02')
		e3 = tf.expand_dims(e2,axis = 1, name = 'expdim03')
		return tf.multiply(e3,self.bceloss, name = 'lossW')
	

	def _define_saver_summary(self, summary = True):
		""" Create Summary and Saver
		Args:
			logdir_train		: Path to train summary directory
			logdir_test		: Path to test summary directory
		"""
		if (self.logdir_train == None) or (self.logdir_test == None):
			raise ValueError('Train/Test directory not assigned')
		else:
			with tf.device(self.cpu):
				self.saver = tf.train.Saver()
			if summary:
				with tf.device(self.gpu):
					self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
					self.test_summary = tf.summary.FileWriter(self.logdir_test)
					#self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
	
	def _init_weight(self):
		""" Initialize weights
		"""
		print('Session initialization')
		self.Session = tf.Session()
		t_start = time.time()
		self.Session.run(self.init)
		print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')
	
	def _init_session(self):
		""" Initialize Session
		"""
		print('Session initialization')
		t_start = time.time()
		self.Session = tf.Session()
		print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')
		
	def _graph_hourglass(self, inputs):
		"""Create the Network
		Args:
			inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3)
		"""
		with tf.name_scope('model'):
			# Storage Table
			# inp = [None] * self.nStack
			hg = [None] * self.nStack
			ll = [None] * self.nStack
			ll_ = [None] * self.nStack
			# drop = [None] * self.nStack
			out = [None] * self.nStack
			out_ = [None] * self.nStack
			sum_ = [None] * self.nStack
			with tf.name_scope('preprocessing'):
				# Input Dim : nbImages x 256 x 256 x 3
				pad1 = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]], name='pad_1')
				# Dim pad1 : nbImages x 260 x 260 x 3
				conv1 = self._conv_bn_relu(pad1, filters= 64, kernel_size = 7, strides = 2, name = 'conv_256_to_128')
				# Dim conv1 : nbImages x 128 x 128 x 64
				r1 = self._residual(conv1, numOut = 128, name = 'r1')
				# Dim pad1 : nbImages x 128 x 128 x 128
				pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
				# Dim pool1 : nbImages x 64 x 64 x 128
				r2 = self._residual(pool1, numOut= 128, name = 'r2')
				r3 = self._residual(r2, numOut= self.nFeat, name = 'r3')
				sum_[0] = r3

			with tf.name_scope('stacks'):
				for i in range(self.nStack):
					with tf.name_scope('stage_' + str(i)):
						hg[i] = self._hourglass(sum_[i], self.nLow, self.nFeat, 'hourglass')
						hgr = self._residual(hg[i], numOut=self.nFeat, name='hgres')
						ll[i] = self._conv_bn_relu(hgr, self.nFeat, 1, 1, 'VALID', name= 'conv')		
						out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
						# out[i] = self._deconv(ll[i], self.outDim, 3, 2, 'SAME', 'out')
						if i < self.nStack - 1:
							ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
							out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
							sum_[i+1] = tf.add_n([out_[i], sum_[i], ll_[i]], name= 'merge')
							# pool = tf.contrib.layers.max_pool2d(out[i], [2,2], [2,2], padding='VALID')
							# out_[i] = self._conv(pool, self.nFeat, 1, 1, 'VALID', 'out_')
							# sum_[i+1] = tf.add_n([out_[i], sum_[i], ll_[i]], name = 'merge')
							
			return tf.stack(out, axis= 1 , name = 'final_output')		
						
				
	def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
		""" Spatial Convolution (CONV2D)
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
		with tf.name_scope(name):
			# Kernel for convolution, Xavier Initialisation
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return conv
	
	def _deconv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
		""" Spatial Deconvolution (DECONV2D)
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME)
			name			: Name of the block
		Returns:
			deconv			: Output Tensor (Deconvolved Input)
		"""
		with tf.name_scope(name):
			deconv = tf.contrib.layers.conv2d_transpose(inputs, filters, [kernel_size,kernel_size], [strides,strides], padding=pad, data_format='NHWC', activation_fn=None)
			return deconv

	def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):
		""" Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			norm			: Output Tensor
		"""
		with tf.name_scope(name):
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm

	
	def _conv_block(self, inputs, numOut, name = 'conv_block'):
		""" Convolutional Block
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""
		with tf.name_scope(name):
			with tf.name_scope('norm_1'):
				norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
				conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
			with tf.name_scope('norm_2'):
				norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
				pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
				conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
			with tf.name_scope('norm_3'):
				norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
				conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
			return conv_3
	
	def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
		""" Skip Layer
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the bloc
		Returns:
			Tensor of shape (None, inputs.height, inputs.width, numOut)
		"""
		with tf.name_scope(name):
			if inputs.get_shape().as_list()[3] == numOut:
				return inputs
			else:
				conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
				return conv				
	
	def _residual(self, inputs, numOut, name = 'residual_block'):
		""" Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			convb = self._conv_block(inputs, numOut)
			skipl = self._skip_layer(inputs, numOut)
			return tf.add_n([convb, skipl], name = 'res_block')
	
	def _hourglass(self, inputs, n, numOut, name = 'hourglass'):
		""" Hourglass Module
		Args:
			inputs	: Input Tensor
			n		: Number of downsampling step
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			# Upper Branch
			up_1 = self._residual(inputs, numOut, name = 'up_1')
			# Lower Branch
			low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
			low_1= self._residual(low_, numOut, name = 'low_1')
			
			if n > 1:
				low_2 = self._hourglass(low_1, n-1, numOut, name = 'low_2')
			else:
				low_2 = self._residual(low_1, numOut, name = 'low_2')
				
			low_3 = self._residual(low_2, numOut, name = 'low_3')
			up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name = 'upsampling')

			return tf.add_n([up_2,up_1], name='out_hg')
	
	def _accuracy_computation(self):
		""" Computes accuracy tensor
		"""
		self.joint_accur = []
		for i in self.accIdx:
			da = self._distAccuracy(self.output[:, self.nStack - 1, :, :,i], self.gtMaps[:, self.nStack - 1, :, :, i], self.batchSize)
			self.joint_accur.append(da)


	def _distAccuracy(self, pred, gtMap, num_image):
		thr = tf.to_float(0.5)
		n, h, w = pred.get_shape().as_list()
		normalize = tf.to_float(w) / 10

		resh_pred = tf.reshape(pred, [num_image, -1])
		argmax_pred = tf.argmax(resh_pred, axis=1)
		preds = tf.to_float(tf.stack([argmax_pred % w, argmax_pred // w], axis=0))

		resh_gt = tf.reshape(gtMap, [num_image, -1])
		argmax_gt = tf.argmax(resh_gt, axis=1)
		gts = tf.to_float(tf.stack([argmax_gt % w, argmax_gt // w], axis=0))
		dists = tf.sqrt(tf.square(preds[0]-gts[0])+tf.square(preds[1]-gts[1])) / normalize
		mask = tf.less_equal(dists, tf.tile([thr], [num_image]))
		# sum_less = tf.reduce_sum(tf.multiply(dists, tf.to_float(mask)))
		# sum_all = tf.reduce_sum(dists)
		sum_less = tf.reduce_sum(tf.to_float(mask))
		sum_all = tf.to_float(num_image)
		return tf.divide(sum_less, sum_all)


	def _graph_hourglass_mobile(self, inputs):
		"""Create the Network
		Args:
			inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
		"""
		# Storage Table
		# inp = [None] * self.nStack
		hg = [None] * self.nStack
		ll = [None] * self.nStack
		ll_ = [None] * self.nStack
		# drop = [None] * self.nStack
		out = [None] * self.nStack
		out_ = [None] * self.nStack
		sum_ = [None] * self.nStack
		with tf.name_scope('model'):
			with tf.name_scope('preprocessing'):
				ks = 3
				pd = int((ks-1)/2)
				# Input Dim : nbImages x 256 x 256 x 3
				pad1 = tf.pad(inputs, [[0,0],[pd,pd],[pd,pd],[0,0]], name='pad_1')
				# Dim pad1 : nbImages x 260 x 260 x 3
				conv1 = self._conv_bn_relu(pad1, filters= 32, kernel_size = ks, strides = 2, name = 'conv_256_to_128')
				# Dim conv1 : nbImages x 128 x 128 x 64
				r1 = self._residual_mobile(conv1, numOut = 32, multi = 2, name = 'r1')
				# Dim pad1 : nbImages x 128 x 128 x 128
				pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
				# Dim pool1 : nbImages x 64 x 64 x 128
				r2 = self._residual_mobile(pool1, numOut= int(self.nFeat), multi = self.multi, name = 'r2')
				r3 = self._residual_mobile(r2, numOut= self.nFeat, multi = self.multi, name = 'r3')
				sum_[0] = r3
			
			with tf.name_scope('stacks'):
				for i in range(self.nStack):
					with tf.name_scope('stage_' + str(i)):
						hg[i] = self._hourglass_mobile(sum_[i], self.nLow, self.nFeat, 'hourglass')
						hgr = self._residual_mobile(hg[i], numOut=self.nFeat, name='hgres')
						ll[i] = self._conv_bn_relu(hgr, self.nFeat, 1, 1, 'VALID', name= 'conv')			
						out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')	
						# out[i] = self._deconv(ll[i], self.outDim, 3, 2, 'SAME', 'out')		
						if i < self.nStack-1:
							ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
							out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
							sum_[i+1] = tf.add_n([out_[i], sum_[i], ll_[i]], name= 'merge')
							# pool = tf.contrib.layers.max_pool2d(out[i], [2,2], [2,2], padding='VALID')
							# out_[i] = self._conv(pool, self.nFeat, 1, 1, 'VALID', 'out_')
							# sum_[i+1] = tf.add_n([out_[i], sum_[i], ll_[i]], name = 'merge')
							
			return tf.stack(out, axis= 1 , name = 'final_output')
	
	def _residual_mobile(self, inputs, numOut, multi=6, name = 'residual_block'):
		""" Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			convb = self._conv_block_mobile(inputs, numOut, multi)
			skipl = self._skip_layer(inputs, numOut)

			return tf.add_n([convb, skipl], name = 'res_block')
	
	
	def _conv_depthwise(self, inputs, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_dw' ):
		""" Depth wise convolution
		Args have the same meaning with function _conv
		"""
		with tf.name_scope(name):
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size,inputs.get_shape().as_list()[3],1]), name='weights')
			conv = tf.nn.depthwise_conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return conv
	
	def _conv_block_mobile(self, inputs, numOut, multi, name = 'conv_block'):
		""" Convolutional Block Based on MobileNet V2
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""
		with tf.name_scope(name):
			with tf.name_scope('norm_1'):
				conv_1 = self._conv(inputs, multi*inputs.get_shape().as_list()[3], kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
				norm_1 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
			with tf.name_scope('norm_2'):
				pad = tf.pad(norm_1, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
				conv_2 = self._conv_depthwise(pad, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
				norm_2 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
			with tf.name_scope('norm_3'):
				conv_3 = self._conv(norm_2, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
				norm_3 = tf.contrib.layers.batch_norm(conv_3, 0.9, epsilon=1e-5, activation_fn = None, is_training = self.training)
			return norm_3

	def _hourglass_mobile(self, inputs, n, numOut, name = 'hourglass'):
		""" Hourglass Module Based on MoblieNet V2
		Args:
			inputs	: Input Tensor
			n		: Number of downsampling step
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			# Upper Branch
			up_1 = self._residual_mobile(inputs, numOut, self.multi, name = 'up_1')
			# Lower Branch
			low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
			low_1= self._residual_mobile(low_, numOut, self.multi, name = 'low_1')
			
			if n > 1:
				low_2 = self._hourglass_mobile(low_1, n-1, numOut, name = 'low_2')
			else:
				low_2 = self._residual_mobile(low_1, numOut, self.multi, name = 'low_2')
				
			low_3 = self._residual_mobile(low_2, numOut, self.multi, name = 'low_3')
			up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name = 'upsampling')

			return tf.add_n([up_2,up_1], name='out_hg')