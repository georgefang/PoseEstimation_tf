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
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import math

class DataGenerator():
	""" DataGenerator Class : To generate Train, Validatidation and Test sets
	for the Human Pose Estimation Model 
	Formalized DATA:
		Inputs:
			Inputs have a shape of (Number of Image) X (Height: 256) X (Width: 256) X (Channels: 3)
		Outputs:
			Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: 64) X (Width: 64) X (OutputDimendion: 16)
	Joints:
		We use the MPII convention on joints numbering
		List of joints:
			00 - Right Ankle
			01 - Right Knee
			02 - Right Hip
			03 - Left Hip
			04 - Left Knee
			05 - Left Ankle
			06 - Pelvis (Not present in other dataset ex : LSP)
			07 - Thorax (Not present in other dataset ex : LSP)
			08 - Neck
			09 - Top Head
			10 - Right Wrist
			11 - Right Elbow
			12 - Right Shoulder
			13 - Left Shoulder
			14 - Left Elbow
			15 - Left Wrist
	
	How to generate Dataset:
		Create a TEXT file with the following structure:
			image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
			[LETTER]:
				One image can contain multiple person. To use the same image
				finish the image with a CAPITAL letter [A,B,C...] for 
				first/second/third... person in the image
 			joints : 
				Sequence of x_p y_p (p being the p-joint)
				/!\ In case of missing values use -1
				
	The Generator will read the TEXT file to create a dictionnary
	Then 2 options are available for training:
		Store image/heatmap arrays (numpy file stored in a folder: need disk space but faster reading)
		Generate image/heatmap arrays when needed (Generate arrays while training, increase training time - Need to compute arrays at every iteration) 
	"""
	def __init__(self, joints_name = None, img_dir=None, train_data_file = None, input_res = 256):
		""" Initializer
		Args:
			joints_name			: List of joints condsidered
			img_dir				: Directory containing every images
			train_data_file		: Text file with training set data
			input_res   		: input image size of hg networks
		"""
		if joints_name == None:
			self.joints_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
		else:
			self.joints_list = joints_name
		self.flipRef = np.array([6,5,4,3,2,1,7,8,9,10,16,15,14,13,12,11])-1
		
		self.letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
		self.img_dir = img_dir
		self.train_data_file = train_data_file
		self.images = os.listdir(img_dir)
		self.input_res = input_res
	
	# --------------------Generator Initialization Methods ---------------------
	
	
	def _create_train_table(self):
		""" Create Table of samples from TEXT file
		"""
		self.train_table = []
		self.data_dict = {}
		input_file = open(self.train_data_file, 'r')
		print('READING TRAIN DATA')
		for line in input_file:
			line = line.strip()
			line = line.split(' ')
			name = line[0]
			box = list(map(int,line[1:5]))

			center = list(map(int, line[5:7]))
			scale = list(map(float, line[7:8]))
			joints = list(map(int, line[8:]))

			if joints != [-1] * len(joints):
				joints = np.reshape(joints, (-1,2))
				w = [1] * joints.shape[0]
				for i in range(joints.shape[0]):
					if np.array_equal(joints[i], [-1,-1]):
						w[i] = 0
				self.data_dict[name] = {'box' : box, 'joints' : joints, 'weights' : w, 'center' : center, 'scale' : scale}			
				self.train_table.append(name)
		input_file.close()
	

	def _randomize(self):
		""" Randomize the set
		"""
		random.shuffle(self.train_table)
	
	def _complete_sample(self, name):
		""" Check if a sample has no missing value
		Args:
			name 	: Name of the sample
		"""
		for i in range(self.data_dict[name]['joints'].shape[0]):
			if np.array_equal(self.data_dict[name]['joints'][i],[-1,-1]):
				return False
		return True

	
	def _create_sets(self, validation_rate = 0.1):
		""" Select Elements to feed training and validation set 
		Args:
			validation_rate		: Percentage of validation data 
		"""
		sample = len(self.train_table)
		valid_sample = int(sample * validation_rate)
		self.train_set = self.train_table[:sample - valid_sample]
		self.valid_set = []
		preset = self.train_table[sample - valid_sample:]
		print('START SET CREATION')
		for elem in preset:
			if self._complete_sample(elem):
				self.valid_set.append(elem)
			else:
				self.train_set.append(elem)
		print('SET CREATED')
		np.save('Dataset-Validation-Set', self.valid_set)
		np.save('Dataset-Training-Set', self.train_set)
		print('--Training set :', len(self.train_set), ' samples.')
		print('--Validation set :', len(self.valid_set), ' samples.')
	
	def generateSet(self, rand = False):
		""" Generate the training and validation set
		Args:
			rand : (bool) True to shuffle the set
		"""
		self._create_train_table()
		if rand:
			self._randomize()
		self._create_sets()
	
	# ---------------------------- Generating Methods --------------------------	
	
	
	def _makeGaussian(self, height, width, sigma = 3, center=None):
		""" Make a square gaussian kernel.
		size is the length of a side of the square
		sigma is full-width-half-maximum
		"""

		gaussian = np.zeros((height, width), dtype=np.float32)
		if center is None:
			x0 = width // 2
			y0 = height // 2
		else:
			x0 = int(center[0])
			y0 = int(center[1])
		d = 0.25 * (2*sigma+1)
		size = int(sigma)
		for i in range(-size, size+1):
			if y0+i >= height or y0+i < 0:
				continue
			for j in range(-size, size+1):
				if x0+j >= width or x0+j < 0:
					continue
				gaussian[int(y0+i),int(x0+j)] = math.exp(-(i**2+j**2)/(2*d**2))
		return gaussian
	
	def _generate_hm(self, height, width ,joints, maxlenght, weight):
		""" Generate a full Heap Map for every joints in an array
		Args:
			height			: Wanted Height for the Heat Map
			width			: Wanted Width for the Heat Map
			joints			: Array of Joints
			maxlenght		: Lenght of the Bounding Box
		"""
		num_joints = joints.shape[0]
		hm = np.zeros((height, width, num_joints), dtype = np.float32)
		for i in range(num_joints):
			if not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1:
				s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
				hm[:,:,i] = self._makeGaussian(height, width, sigma= s, center= (joints[i,0], joints[i,1]))
			else:
				hm[:,:,i] = np.zeros((height,width))
		return hm
		
	def _crop_data(self, height, width, box, joints, boxp = 0.05):
		""" Automatically returns a padding vector and a bounding box given
		the size of the image and a list of joints.
		Args:
			height		: Original Height
			width		: Original Width
			box			: Bounding Box
			joints		: Array of joints
			boxp		: Box percentage (Use 20% to get a good bounding box)
		"""
		padding = [[0,0],[0,0],[0,0]]
		j = np.copy(joints)
		if box[0:2] == [-1,-1]:
			j[joints == -1] = 1e5
			box[0], box[1] = min(j[:,0]), min(j[:,1])
		crop_box = [box[0] - int(boxp * (box[2]-box[0])), box[1] - int(boxp * (box[3]-box[1])), box[2] + int(boxp * (box[2]-box[0])), box[3] + int(boxp * (box[3]-box[1]))]
		if crop_box[0] < 0: crop_box[0] = 0
		if crop_box[1] < 0: crop_box[1] = 0
		if crop_box[2] > width -1: crop_box[2] = width -1
		if crop_box[3] > height -1: crop_box[3] = height -1
		new_h = int(crop_box[3] - crop_box[1])
		new_w = int(crop_box[2] - crop_box[0])
		crop_box = [crop_box[0] + new_w //2, crop_box[1] + new_h //2, new_w, new_h]
		if new_h > new_w:
			bounds = (crop_box[0] - new_h //2, crop_box[0] + new_h //2)
			if bounds[0] < 0:
				padding[1][0] = abs(bounds[0])
			if bounds[1] > width - 1:
				padding[1][1] = abs(width - bounds[1])
		elif new_h < new_w:
			bounds = (crop_box[1] - new_w //2, crop_box[1] + new_w //2)
			if bounds[0] < 0:
				padding[0][0] = abs(bounds[0])
			if bounds[1] > height - 1:
				padding[0][1] = abs(height - bounds[1])
		crop_box[0] += padding[1][0]
		crop_box[1] += padding[0][0]
		return padding, crop_box
	
	def _adjust_scale(self, scale):
		s = 0.25
		rnd = max(-2*s, min(2*s, s * np.random.standard_normal()))
		scale[0] = scale[0] * (2**rnd)
		return scale

	def _crop_data_cs(self, height, width, center, scale):
		rp = 100
		padding = [[0,0],[0,0],[0,0]]
		length = int(rp * scale[0])
		box = [int(center[0] - length), int(center[1] - length), int(center[0] + length), int(center[1] + length)]
		for i in range(2):
			if box[i] < 0:
				padding[0][i] = 0 - box[i]
				# box[i] = 0
		if box[2] > width - 1:
			padding[1][0] = box[2] - width + 1
			# box[2] = width - 1
		if box[3] > height - 1:
			padding[1][1] = box[3] - height + 1
			# box[3] = height - 1
		return padding, box
	
	def _relative_joints_pos(self, box, padding, joints, to_size = 64):
		new_j = np.copy(joints)
		new_j = new_j - [box[0], box[1]]
		new_j = new_j * to_size / (box[2] - box[0] + 0.0000001)
		return new_j

	def _crop_image(self, img, pad, box):
		image = np.zeros((box[3]-box[1]+1, box[2]-box[0]+1, 3), dtype=np.uint8)
		# print("image shape is {0} {1} {2}".format(image.shape[0], image.shape[1], image.shape[2]))
		assert image.shape[0] == image.shape[1]
		leng = image.shape
		image[pad[0][1]:leng[0]-pad[1][1], pad[0][0]:leng[1]-pad[1][0]] = img[box[1]+pad[0][1]:box[3]-pad[1][1]+1, box[0]+pad[0][0]:box[2]-pad[1][0]+1]

		return image 

	def _crop_img(self, img, padding, crop_box):
		""" Given a bounding box and padding values return cropped image
		Args:
			img			: Source Image
			padding	: Padding
			crop_box	: Bounding Box
		"""
		img = np.pad(img, padding, mode = 'constant')
		max_lenght = max(crop_box[2], crop_box[3])
		img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
		return img
		
	def _crop(self, img, hm, padding, crop_box):
		""" Given a bounding box and padding values return cropped image and heatmap
		Args:
			img			: Source Image
			hm			: Source Heat Map
			padding	: Padding
			crop_box	: Bounding Box
		"""
		img = np.pad(img, padding, mode = 'constant')
		hm = np.pad(hm, padding, mode = 'constant')
		max_lenght = max(crop_box[2], crop_box[3])
		img = img[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght //2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght //2]
		hm = hm[crop_box[1] - max_lenght //2:crop_box[1] + max_lenght//2, crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
		return img, hm
	
	def _relative_joints(self, box, padding, joints, to_size = 64):
		""" Convert Absolute joint coordinates to crop box relative joint coordinates
		(Used to compute Heat Maps)
		Args:
			box			: Bounding Box 
			padding	: Padding Added to the original Image
			to_size	: Heat Map wanted Size
		"""
		new_j = np.copy(joints)
		max_l = max(box[2], box[3])
		new_j = new_j + [padding[1][0], padding[0][0]]
		new_j = new_j - [box[0] - max_l //2,box[1] - max_l //2]
		new_j = new_j * to_size / (max_l + 0.0000001)
		return new_j.astype(np.int32)
		
		
	def _augment(self,img, hm, max_rotation = 30):
		""" # TODO : IMPLEMENT DATA AUGMENTATION 
		"""
		if random.choice([0,1]): 
			r_angle = np.random.randint(-1*max_rotation, max_rotation)
			img = 	transform.rotate(img, r_angle, preserve_range = True)
			hm = transform.rotate(hm, r_angle)
		return img, hm
	
	def _augmentIJ(self, img, joints, max_rotation = 30, hm_size=64):
		new_j = np.copy(joints)
		if np.random.uniform() > 0.6:
			r_angle = max(-2*max_rotation, min(2*max_rotation, max_rotation*np.random.standard_normal()))
			img = transform.rotate(img, r_angle, preserve_range = True)
			rad = r_angle * math.pi / 180
			rotm = [[math.cos(rad), -math.sin(rad)],[math.sin(rad), math.cos(rad)]]
			mid = np.tile([[hm_size/2-0.5, hm_size/2-0.5]], [joints.shape[0], 1])
			new_j = np.dot(new_j-mid, rotm) + mid
		return img, new_j

	def _movedown_scale(self, img, box, move=15, scale=1.25):
		xsize = box[2] - box[0]
		ysize = box[3] - box[1]

	def _flip_data(self, img, hm):	
		img = cv2.flip(img, 1)
		for i in np.arange(hm.shape[2]):
			hm[:,:,i] = cv2.flip(hm[:,:,i], 1)
		hm = hm[:,:,self.flipRef]
			
		return img, hm

	def _mul_frac(self, img):
		for i in np.arange(img.shape[2]):
			img = img * np.random.uniform(0.6,1.4)
		img[img<0] = 0
		img[img>255] = 255
		return img

	def _move_horizon(self, img, joints):
		if np.random.uniform() > 0.5:
			new_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]), np.uint8)
			p = np.random.uniform() * 0.2 * img.shape[1]
			pi = int(p)
			pj = int(p * 64 / img.shape[0])
			if p > 0:
				new_img[:, :img.shape[1]-pi] = img[:, pi:img.shape[1]]
			else:
				new_img[:, pi:] = img[:, :img.shape[1]-pi]

			new_j = joints - np.tile([[pj, 0]], [joints.shape[0], 1])
		else:
			new_img = np.copy(img)
			new_j = np.copy(joints)
		return new_img, new_j


	# ----------------------- Batch Generator ----------------------------------

	def _sam_generator(self, batch_list, batch_size = 16, stacks = 4, normalize = True, sample_set = 'train'):
		""" Sample Generator
		Args:
			See Args section in self._generator
		"""
		ires = self.input_res
		ores = 64
		train_img = np.zeros((batch_size, ires, ires,3), dtype = np.float32)
		train_gtmap = np.zeros((batch_size, stacks, ores, ores, len(self.joints_list)), np.float32)
		train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
		i = 0
		while i < batch_size:
			# try:
				# if sample_set == 'train':
				# 	name = random.choice(self.train_set)
				# elif sample_set == 'valid':
				# 	name = random.choice(self.valid_set)
			if sample_set == 'train':
				name = self.train_set[batch_list[i]]
			elif sample_set == 'valid':
				name = self.valid_set[batch_list[i]]
			joints = np.copy(self.data_dict[name]['joints'])
			box = np.copy(self.data_dict[name]['box'])
			center = np.copy(self.data_dict[name]['center'])
			scale = np.copy(self.data_dict[name]['scale'])
			weight = np.asarray(self.data_dict[name]['weights'])
			train_weights[i] = weight 
			img = self.open_img(name, color='BGR')
			center[1] = center[1] + int(15 * scale[0])
			scale[0] = scale[0] * 1.25
			if sample_set == 'train':
				scale = self._adjust_scale(scale)
			padd, cbox = self._crop_data_cs(img.shape[0], img.shape[1], center, scale)
			new_j = self._relative_joints_pos(cbox,padd, joints, to_size=ores)
			img = self._crop_image(img, padd, cbox)
			# img = img.astype(np.uint8)
			img = cv2.resize(img, (ires,ires))
			if sample_set == 'train':
				img, new_j = self._augmentIJ(img, new_j)
				img, new_j = self._move_horizon(img, new_j)
			hm = self._generate_hm(ores, ores, new_j, ores, weight)
			if sample_set == 'train':
				if np.random.uniform() > 0.5:
					img, hm = self._flip_data(img, hm)
				img = self._mul_frac(img)
			hm = np.expand_dims(hm, axis = 0)
			hm = np.repeat(hm, stacks, axis = 0)
			if normalize:
				train_img[i] = img.astype(np.float32) / 255
			else :
				train_img[i] = img.astype(np.float32)
			train_gtmap[i] = hm
			i = i + 1
			# except :
				# print('error file: ', name)			

		return train_img, train_gtmap, train_weights


	def epochsize_cat(self, epoch, batch, sample = 'train'):
		if sample == 'train':
			length = len(self.train_set)
		elif sample == 'valid':
			length = len(self.valid_set)
		ebsize = epoch * batch
		randlist = np.random.permutation(length)
		times = int(ebsize/length)
		for i in np.arange(0, times):
			randlist = np.hstack( (randlist, np.random.permutation(length)) )
		return randlist[:ebsize]
	
	# ---------------------------- Image Reader --------------------------------				
	def open_img(self, name, color = 'RGB'):
		""" Open an image 
		Args:
			name	: Name of the sample
			color	: Color Mode (RGB/BGR/GRAY)
		"""
		if name[-1] in self.letter:
			name = name[:-1]
		img = cv2.imread(os.path.join(self.img_dir, name))
		if color == 'RGB':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img
		elif color == 'BGR':
			return img
		elif color == 'GRAY':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')
	
				
		
		