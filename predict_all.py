import sys
sys.path.append('./')

import time
import numpy as np
import tensorflow as tf
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import threading
import cv2
import os
import glob
import h5py
import imageio
from xml.dom import minidom
from datagen import DataGenerator
from utils2d import show_joints
import inference



class PredictAll():

	#-------------------------INITIALIZATION METHODS---------------------------
	def __init__(self, model, resize=False, hm=True):
		self.model = model
		self.resize = resize
		self.hm = hm
		self.originSize = False
		self.standard = False
		self.p3d = False
		self.show3d = False
		self.joint2dNum = 16
		self.joint3dNum = 17
		

	def set_originSize(self, orig):
		self.originSize = orig

	def set_standardSize(self, size, std=True):
		self.standard = std
		self.stdSize = size

	def add_model3d(self, model3d, show3d):
		self.model3d = model3d
		self.p3d = True
		self.show3d = show3d
		self.joint3dNames = ['hips1', 'rightUpLeg', 'rightLeg', 'rightFoot', 'leftUpLeg', 'leftLeg', 'leftFoot', \
			'spine1', 'neck', 'head', 'head1', 'leftArm', 'leftForearm', 'leftHand', 'rightArm', 'rightForearm', 'rightHand']


	def predict_general( self, img, wt=0):
		imgOrig = np.copy(img)
		if self.resize:
			img = cv2.resize(img, (256,256))
		else:
			img, scale, _ = self.model.preProcessImage(img)
		test_img = img

		test_img = test_img.astype(np.float32) / 255
		startTime = time.time()
		joints, hms = self.model.predictJointsFromImageByMean(test_img)
		print('predict time is: ', 1000*(time.time()-startTime), 'ms')
		test_img = test_img * 255
		test_img = test_img.astype(np.uint8)

		if self.standard:
			joints = joints * self.stdSize / 256.0
			imgStd = cv2.resize(test_img, (self.stdSize, self.stdSize))
			imgPred = show_joints(imgStd, joints, wt=wt, name='std')

		if self.originSize:
			origShape = imgOrig.shape[0:2]
			msize = np.amax(origShape)
			if self.resize:
				joints[:,0] = joints[:,0] * origShape[1] / self.model.params['img_size']
				joints[:,1] = joints[:,1] * origShape[0] / self.model.params['img_size']
				imgPred = show_joints(imgOrig, joints, wt=wt, name='orgin')
			else:
				scale = float(msize) / self.model.params['img_size']
				jointsOri = (joints+0.5) * scale
				if origShape[0] < origShape[1]:
					jointsOri[:,1] = jointsOri[:,1] - (origShape[1]-origShape[0])/2
				else:
					jointsOri[:,0] = jointsOri[:,0] - (origShape[0]-origShape[1])/2
				imgPred = show_joints(imgOrig, jointsOri, wt=wt, name='orgin')

		elif self.hm:
			imgPred = show_joints(test_img, joints, hms, wt)
		else:
			imgPred = show_joints(test_img, joints, wt=wt)

		return joints, imgPred, scale

	def predict_image( self, imgname):
		img = cv2.imread(imgname)
		pose2d, _, _ = self.predict_general( img )
		if self.p3d:
			pose3d = self.model3d.predict_3dpose(2*pose2d)
			channels = pose3d[0,:]
			plt.ion()
			channels[1::3] = -channels[1::3]
			channels[2::3] = -channels[2::3]
			self.pose_in_plot( plt.figure(), channels, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True)
			cv2.waitKey(0)

	def predict_h36mImages( self, imgpath):
		# subjects = [1,5,6,7,8,9,11]
		subjects = [9,11]
		tag = 'ground_truth'
		for sub in subjects:
			print('Reading subject {}'.format(sub))
			subpath = os.path.join(imgpath, 'S{}'.format(sub))
			groundpath = os.path.join(subpath, 'ground_truth_h5')
			imagepath = os.path.join(subpath, 'ImagesCrop')
			for gname in os.listdir(groundpath):
				print(gname)
				with h5py.File( os.path.join(groundpath, gname), 'r') as h5f:
					ground = h5f[tag][:]
				basename = gname[:-3]
				fnum = ground.shape[0]
				for i in np.arange(fnum):
					imgname = os.path.join(imagepath, basename, basename+'_{}.jpg'.format(i+1))
					img = cv2.imread(imgname)
					joints, _, scale = self.predict_general(img, wt=10)
					if cv2.waitKey(1) == 27:
				 		cv2.destroyAllWindows()
				 		return

	def predict_camera( self, camidx):
		cam = cv2.VideoCapture(camidx)
		while True:
			# startTime = time.time()
			ret_val, img = cam.read()
			print(img.shape)
		 	img = cv2.flip(img, 1)
		 	self.predict_general(img, wt=10)
		 	if cv2.waitKey(1) == 27:
		 		cv2.destroyAllWindows()
		 		break

	def predict_video( self, videoname, savename ):
		stop = False
		save = False
		while True:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			if savename is not None:
				save = True
				savefile = cv2.VideoWriter(savename, fourcc, 30.0, (544, 624))

			video = cv2.VideoCapture(videoname)
			if not video.isOpened():
				print("fail to read video file: {}".format(videoname))
				return
			if self.show3d:
				fig = plt.figure()
				plt.ion()
			jointsQueue = []
			while True:
				ret, frame = video.read()
				if ret == True:
					shape = frame.shape
					frame = frame[int(0.2*shape[0]):int(0.85*shape[0])]
					joints, framePred, scale = self.predict_general(frame, wt=10)
					if save:
						savefile.write(framePred)
					if self.p3d:
						joints3d = self.model3d.predict_3dpose(2*joints)
						jointsQueue.append(joints3d)
						if self.show3d:
							joints3dShow = np.copy(joints3d)
							joints3dShow[:, 1::3] = -joints3dShow[:, 1::3]
							joints3dShow[:, 2::3] = -joints3dShow[:, 2::3]
							self.pose_in_plot(fig, joints3dShow[0], lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True)
					if cv2.waitKey(1) == 27:
						stop = True
						break
				elif save or self.p3d:
					stop = True
					break

			if self.p3d:
				jointsQueue = np.vstack(jointsQueue)
				sname = videoname[:-4] + '.xml'
				self.save_pose_data_xml(jointsQueue, sname, dataroot=None, dim=3, filter=4)
			
			if save or stop:
				video.release()
				if save:
					savefile.release()
				cv2.destroyAllWindows()
				return

	def predict_video_imageio( self, videoname, savename ):
		stop = False
		save = False
		while True:
			fourcc = cv2.VideoWriter_fourcc(*'MP4V')
			if savename is not None:
				save = True
				savefile = cv2.VideoWriter(savename, fourcc, 30.0, (256, 256))

			if self.show3d:
				fig = plt.figure()
				plt.ion()
			jointsQueue = []
			while True:
				# ret, frame = video.read()
				video = imageio.get_reader(videoname, 'ffmpeg')
				for num, frame in enumerate(video):
					frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
					shape = frame.shape
					frame = frame[int(0.15*shape[0]):int(0.85*shape[0])]
					joints, framePred, scale = self.predict_general(frame, wt=10)
					if save:
						savefile.write(framePred)
					if self.p3d:
						joints3d = self.model3d.predict_3dpose(2*joints)
						jointsQueue.append(joints3d)
						if self.show3d:
							joints3dShow = np.copy(joints3d)
							joints3dShow[:, 1::3] = -joints3dShow[:, 1::3]
							joints3dShow[:, 2::3] = -joints3dShow[:, 2::3]
							self.pose_in_plot(fig, joints3dShow[0], lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True)
					if cv2.waitKey(1) == 27:
						stop = True
						break
				if save or stop or self.p3d:
					break

			if self.p3d:
				jointsQueue = np.vstack(jointsQueue)
				sname = videoname[:-4] + '.xml'
				self.save_pose_data_xml(jointsQueue, sname, dataroot=None, dim=3, filter=4)

			if save:
				savefile.release()
			return

	# def predict_3d( self, imgpath, imgname):
	# 	img = cv2.imread(imgname)
	# 	pose2d, _, _ = self.predict_general( img, wt=10 )
	# 	pose3d = self.model3d.predict_3dpose(2*pose2d)
	# 	channels = pose3d[0,:]
	# 	fig = plt.figure()
	# 	plt.ion()
	# 	self.pose_in_plot( fig, channels, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True)
	# 	print('predict done')
	# 	cv2.waitKey(0)

	def pose_in_plot(self, fig, channels, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
		fig.clf()
		fig.suptitle("3d pose", fontsize=12)

		ax = fig.add_subplot(111, projection='3d')
		# assert channels.size == 17*3, "channels should have 51 entries, it has %d instead" % channels.size
		temp = np.copy(channels[2::3])
		channels[2::3] = channels[1::3]
		channels[1::3] = -temp
		vals = np.reshape( channels, (17, -1) )
		I   = np.array([1,2,3,1,5,6,1, 8, 9,10, 9,12,13, 9,15,16])-1 # start points
		J   = np.array([2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16,17])-1 # end points
		LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

		# Make connection matrix
		for i in np.arange( len(I) ):
			x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
			ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
		RADIUS = 750 # space around the subject
		xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
		ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
		ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
		ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

		if add_labels:
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.set_zlabel("z")
		plt.pause(0.005)


	def save_pose_data_xml(self, data, name, dataroot=None, dim=3, filter=None):
		data= np.reshape(data, (data.shape[0], -1))
		dnum = data.shape[0]
		jnum = data.shape[1]/dim

		if filter:
			datatemp = np.copy(data)
			for i in range(dnum):
				den = filter
				if i<filter:
					den = i
				elif dnum-1-i < filter:
					den = dnum-1-i
				data[i] = np.mean(datatemp[i-den:i+den+1], axis=0)
		if dataroot is not None:
			data = data - np.tile(data[:,:dim], [1, jnum])
			data = data + np.tile(dataroot[:dnum,:dim], [1, jnum]) 
		data[:,1::3] = -data[:,1::3]
		data[:,2::3] = -data[:,2::3]
		impl = minidom.getDOMImplementation()
		doc = impl.createDocument(None, None, None)
		rootElement = doc.createElement('Pose{}d'.format(dim))
		for id in range(dnum):
			frameElement = doc.createElement('frame')
			frameElement.setAttribute('id', str(id))
			for nj in range(jnum):
				jointElement = doc.createElement('joint')
				jointElement.setAttribute('name', self.joint3dNames[nj])
				jointElement.setAttribute('xpos', str(data[id,nj*dim+0]))
				jointElement.setAttribute('ypos', str(data[id,nj*dim+1]))
				jointElement.setAttribute('zpos', str(data[id,nj*dim+2]))
				frameElement.appendChild(jointElement)
			rootElement.appendChild(frameElement)
		doc.appendChild(rootElement)
		f = open(name, 'w')
		doc.writexml(f, addindent='  ', newl='\n')
		f.close()
