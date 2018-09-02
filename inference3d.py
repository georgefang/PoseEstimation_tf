import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from linear_model import LinearModel 



class Inference3d():
	""" 3D Inference Class
	Predict 3D pose from 2d pose
	"""

	def __init__(self, model_dir, params):
		# self.init_model_params(params)
		print('Initializing 3d pose model')
		self.init_model_3d(params)
		print('Finish initialization')
		self.use_root = params['use_root']

		model_file = os.path.join(model_dir, params['model_3d'])
		mean_std_file = os.path.join(model_dir, params['mean_std_3d'])
		
		print('Loading 3d pose model')
		self.load_model_3d(load=model_file)
		print('Finish loading 3d pose model')
		self.load_mean_std(file=mean_std_file)

		self.SH_TO_GT_PERM = np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10])
		self.root2d_init = None

	def init_model_3d(self, params):
		# device_cout = {"GPU": 1}
		# with tf.Session(config=tf.ConfigProto(
		#     device_count=device_count,
		#     allow_soft_placement=True )) as sess:
		self.sess = tf.Session()
		summaries_dir = os.path.join( params['train_dir'], params['summaries_dir'] )
		self.linearModel = LinearModel(params['linear_size'], params['num_layers'], params['residual'], params['batch_norm'], \
			params['max_norm'], params['batch_size'], params['learning_rate'], summaries_dir, params['use_root'], dtype=tf.float32)
		self.linearModel.train_writer.add_graph( self.sess.graph)

	def load_model_3d(self, load):
		self.linearModel.saver.restore(self.sess, load)

	def load_mean_std(self, file):
		"""
		load mean and std data of human pose from h5 file
		these data are from h3.6m dataset
		"""
		with h5py.File(file, 'r') as h5f:
			self.mean_2d = h5f['mean_2d'][:]
			self.std_2d  = h5f['std_2d'][:]
			self.mean_3d = h5f['mean_3d'][:]
			self.std_3d  = h5f['std_3d'][:]
	
	def predict_3dpose(self, data_2d):
		"""
		predict 3d pose from 2d pose
		"""
		data, data_root = self.read_data_input(data_2d)
		data = np.reshape(data, [1, -1])
		data_root = np.reshape(data_root, [1, -1])
		outdim = 17*3 if self.use_root else 16*3
		dec_out = np.zeros((data.shape[0], outdim), dtype=np.float32)

		_, _, pose3d = self.linearModel.step(self.sess, data, dec_out, 1.0, isTraining=False)
		pose3d = self.unNormalizeData(pose3d, self.mean_3d, self.std_3d)
		pose3d[:,3:] = pose3d[:,3:] + np.tile(pose3d[:,:3], [1, int(pose3d.shape[1]/3)-1])
		
		return pose3d

	def read_data_input(self, data_2d):
		"""
		load 2d input data from 2d-pose-predict demo,
		"""
		poses = data_2d[self.SH_TO_GT_PERM, :]
		poses = np.reshape(poses,[-1])
		if self.root2d_init is None:
			self.root2d_init = np.copy(poses[:2])

		data, data_root = self.postprocess_data(poses, dim=2)
		data = self.normalize_data( data, self.mean_2d, self.std_2d)

		return data, data_root

	def normalize_data(self, data, mu, stddev):
		"""
		Normalizes a dictionary of poses

		Args
		data: dictionary where values are
		mu: np vector with the mean of the data
		stddev: np vector with the standard deviation of the data
		Returns
		data_out: dictionary with same keys as data, but values have been normalized
		"""

		data_out = np.divide( (data - mu), stddev )
		if self.use_root:
			return data_out
		else:
			return data_out[2:]


	def unNormalizeData(self, normalized_data, data_mean, data_std):
		"""
		Un-normalizes a matrix whose mean has been substracted and that has been divided by
		standard deviation. Some dimensions might also be missing

		Args
		normalized_data: nxd matrix to unnormalize
		data_mean: np vector with the mean of the data
		data_std: np vector with the standard deviation of the data
		Returns
		orig_data: the input normalized_data, but unnormalized
		"""
		T, D = normalized_data.shape 
		if not self.use_root:
			pad = np.zeros((T,3), dtype=np.float32)
			normalized_data = np.hstack((pad,normalized_data))
			D += 3
		# Multiply times stdev and add the mean
		stdMat = data_std.reshape((1, D))
		stdMat = np.repeat(stdMat, T, axis=0)
		meanMat = data_mean.reshape((1, D))
		meanMat = np.repeat(meanMat, T, axis=0)
		orig_data = np.multiply(normalized_data, stdMat) + meanMat
		return orig_data



	def postprocess_data( self, poses, dim ):
		"""
		Center 3d points around root

		Args
		poses: dictionary with 3d data
		Returns
		poses_set: dictionary with 3d data centred around root (center hip) joint
		root_positions: dictionary with the original 3d position of each pose
		"""
		root_positions = {}
		# for k in poses_set.keys():
		# Keep track of the global position
		root_positions = copy.deepcopy(poses[:dim])

		# Remove the root from the 3d position
		poses = poses - np.tile( poses[:dim], [poses.shape[0]/dim] )
		poses[:dim] = root_positions - self.root2d_init
		poses_set = poses

		return poses_set, root_positions