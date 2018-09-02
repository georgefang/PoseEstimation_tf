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
import sys
sys.path.append('./')

from hourglass import HourglassModel
from time import time, clock
import numpy as np
import tensorflow as tf
import scipy.io
import cv2
from datagen import DataGenerator


class Inference():
	""" Inference Class
	Use this file to make your prediction
	Easy to Use
	Images used for inference should be RGB images (int values in [0,255])
	Methods:

	"""
	def __init__(self, params, model = 'hg_refined_tiny_200'):
		""" Initilize the Predictor
		Args:
			config_file 	 	: *.cfg file with model's parameters
			model 	 	 	 	: *.index file's name. (weights to load) 
		"""
		t = time()
		self.params = params
		self.HG = HourglassModel(params=params, dataset=None, training=False)
		self.graph = tf.Graph()
		self.model_init()
		self.load_model(load = model)
		self._create_prediction_tensor()
		# self.filter = VideoFilters()
		print('Done: ', time() - t, ' sec.')

	
		# ---------------------------MODEL METHODS---------------------------------
	def model_init(self):
		""" Initialize the Hourglass Model
		"""
		t = time()
		with self.graph.as_default():
			self.HG.create_model()
		print('Graph Generated in ', int(time() - t), ' sec.')
	
	def load_model(self, load = None):
		""" Load pretrained weights (See README)
		Args:
			load : File to load
		"""
		with self.graph.as_default():
			self.HG.restore(load)
		

			
	def _create_prediction_tensor(self):
		""" Create Tensor for prediction purposes
		"""
		with self.graph.as_default():
			with tf.name_scope('prediction'):
				self.HG.pred_sigmoid = tf.nn.sigmoid(self.HG.output[:,self.HG.nStack - 1], name= 'sigmoid_final_prediction')
				self.HG.pred_final = self.HG.output[:,self.HG.nStack - 1]
		print('Prediction Tensors Ready!')
	
	
	
	#----------------------------PREDICTION METHODS---------------------------
	
	def pred(self, img, debug = False, sess = None):
		""" Given a 256 x 256 image, Returns prediction Tensor
		This prediction method returns values in [0,1]
		Use this method for inference
		Args:
			img		: Image -Shape (256 x256 x 3) -Type : float32
			debug	: (bool) True to output prediction time
		Returns:
			out		: Array -Shape (64 x 64 x outputDim) -Type : float32
		"""
		if debug:
			t = time()
		if sess is None:
			out = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
		else:
			out = sess.run(self.HG.pred_sigmoid, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})

		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out
	
	def pred_multi(self, imgs, debug = False, sess=None):
		""" Given n * 256 x 256 image, Returns prediction Tensor
		This prediction method returns values in [0,1]
		Use this method for inference
		Args:
			img		: Image -Shape (n * 256 x256 x 3) -Type : float32
			debug	: (bool) True to output prediction time
		Returns:
			out		: Array -Shape (64 x 64 x outputDim) -Type : float32
		"""
		if debug:
			t = time()
		assert len(imgs.shape) == 4
		if sess is None:
			out = self.HG.Session.run(self.HG.output, feed_dict={self.HG.img : imgs})
		else:
			out = sess.run(self.HG.output, feed_dict={self.HG.img : imgs})
		
		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out[:,-1]


	# -----------------------------Image Prediction----------------------------
	def predictJointsFromImage(self, img):
		image = np.copy(img)
		hms = self.pred(image)
		hm = hms[0]
		hmshape = hm.shape
		assert len(hmshape)==3
		joints = np.zeros((hmshape[-1], 2), dtype=np.int64)
		for i in range(0, hmshape[-1]):
			resh = np.reshape(hm[:,:,i], [-1])
			arg = np.argmax(resh)
			#print("hm: {}".format(outhm[:,:,i]))
			joints[i][0] = arg % hmshape[1]
			joints[i][1] = arg // hmshape[1]
			#print("joint {0}: ({1}, {2})".format(i, joints[i][0], joints[i][1]))
			
		joints = joints * image.shape[0] / hmshape[0]
		return joints, hm

	# -----------------------------Image Prediction By Mean Method----------------
	def predictJointsFromImageByMean(self, img):
		image = np.copy(img)
		hms = self.pred(image)
		hm = hms[0]
		hmshape = hm.shape
		assert len(hmshape)==3
		joints = np.zeros((hmshape[-1], 2), dtype=np.float32)
		INDEX = np.arange(hmshape[0])
		sum_all = np.sum(hm, axis=(0,1))
		sum_row = np.sum(hm, axis=0)
		sum_col = np.sum(hm, axis=1)
		joints[:,0] = sum_row.T.dot(INDEX) / sum_all
		joints[:,1] = sum_col.T.dot(INDEX) / sum_all

		joints = (joints+0.5) * image.shape[0] / hmshape[0]
		return joints, hm	

	# -----------------------------Multiple Images Prediction----------------------------
	def predictJointsFromMultiImage(self, imgs):
		# images = np.copy(imgs)
		hms = self.pred_multi(imgs)
		hmshape = hms.shape
		assert len(hmshape)==4
		joints = np.zeros((hmshape[0], hmshape[-1], 2), dtype=np.int64)
		resh = hms.reshape(hmshape[0], -1, hmshape[-1])
		max_idx = np.argmax(resh, axis=1)
		joints[:,:,0] = max_idx % hmshape[2]
		joints[:,:,1] = max_idx // hmshape[2]
		joints = joints * imgs.shape[1] / hmshape[1]
		return joints, hms


	def preProcessImage(self, img):
		""" RETURN THE RESIZE IMAGE WHICH SIZE IS 256*256
		ARGS:
			img: input image
		"""
		shape = img.shape[0:2]
		assert len(shape) == 2
		sizeNor = self.params['img_size']
		msize = np.amax(shape)
		scale = float(sizeNor) / msize
		shape_new = np.array([int(shape[0]*scale), int(shape[1]*scale)])
		imgre = cv2.resize(img, (int(shape_new[1]), int(shape_new[0])))
		imgsq = np.zeros((sizeNor, sizeNor, 3), dtype=np.uint8)
		leftup = [0,0]
		leftup[0] = sizeNor/2 - shape_new[0]/2
		leftup[1] = sizeNor/2 - shape_new[1]/2
		imgsq[leftup[0]:leftup[0]+imgre.shape[0], leftup[1]:leftup[1]+imgre.shape[1], :] = np.copy(imgre) 
		return imgsq, scale, leftup

		
		
		
		