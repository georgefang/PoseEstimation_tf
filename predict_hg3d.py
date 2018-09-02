import os
import time
from inference import Inference
from hourglass import HourglassModel
from datagen import DataGenerator
from utils2d import process_config
import numpy as np
import tensorflow as tf
import configparser
import cv2
from predict_all import PredictAll
from inference3d import Inference3d

tf.app.flags.DEFINE_string("model_dir", "trained/stack2_hg", "pose model directory")
tf.app.flags.DEFINE_string("config_file", "config_mpii.cfg", "config file name")
tf.app.flags.DEFINE_string("model_file", "hg_model_100", "pose model file name")
tf.app.flags.DEFINE_boolean("resize", False, "whether to resize the image to 256*256 directly")
tf.app.flags.DEFINE_boolean("hm", False, "whether to show the heat maps")
tf.app.flags.DEFINE_string("image_file", None, "image file name")
tf.app.flags.DEFINE_integer("camera", None,  "web camera index")
tf.app.flags.DEFINE_string("video", None, "predicted video filename")
tf.app.flags.DEFINE_string("video_save", None, "saved file name of predicted video")
tf.app.flags.DEFINE_string("h36mDir", "../StackedHourglass/data/h36m", "h36m dataset directory")
tf.app.flags.DEFINE_boolean("predict3d", False, "whether to predict 3d pose")
tf.app.flags.DEFINE_string("model3d_dir", "trained/baseline_nr", "3d pose model directory")
tf.app.flags.DEFINE_string("config_file_3d", "config_linear.cfg", "3d config file name")
tf.app.flags.DEFINE_boolean("show3d", False, "whether to show 3d pose result")

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':
	print('--Parsing Config File')

	modeldir = FLAGS.model_dir
	configfile = os.path.join(modeldir, FLAGS.config_file)
	modelfile = os.path.join(modeldir, FLAGS.model_file)
	print(modelfile)

	params = process_config(configfile)
	model = Inference(params=params, model=modelfile)
	predict = PredictAll(model=model, resize=FLAGS.resize, hm=FLAGS.hm)
	predict.set_originSize(orig=True)

	if FLAGS.predict3d:
		configfile3d = os.path.join(FLAGS.model3d_dir, FLAGS.config_file_3d)
		params_3d = process_config(configfile3d)
		model3d = Inference3d(FLAGS.model3d_dir, params_3d)
		predict.add_model3d(model3d, FLAGS.show3d)

	if FLAGS.image_file is not None:
		# single image prediction
		predict.predict_image(FLAGS.image_file)
	elif FLAGS.camera is not None:
		predict.predict_camera(FLAGS.camera)
	elif FLAGS.video is not None:
		predict.predict_video(FLAGS.video, FLAGS.video_save)
		# predict.predict_video_imageio(FLAGS.video, FLAGS.video_save)
	elif FLAGS.h36mDir is not None:
		predict.predict_h36mImages( FLAGS.h36mDir)
	# elif FLAGS.predict3d:
	# 	predict.predict_3d(FLAGS.image_file)