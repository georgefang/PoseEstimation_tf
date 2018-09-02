"""
Pose Estimation
Train Hourglass Networks
"""

from hourglass import HourglassModel
from datagen import DataGenerator
from utils2d import process_config
import tensorflow as tf
import os

tf.app.flags.DEFINE_string("configfile", "config/config_mpii.cfg", "config file name")
tf.app.flags.DEFINE_string("loadmodel", None, "model name used to continue training")

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config( FLAGS.configfile )
	os.system('mkdir -p {}'.format(params['saver_directory']))
	os.system('cp {0} {1}'.format(FLAGS.configfile, params['saver_directory']))
	
	print('--Creating Dataset')
	dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], params['img_size'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel(params=params, dataset=dataset, training=True)
	model.create_model()
	model.do_train(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset=None, load=FLAGS.loadmodel)
