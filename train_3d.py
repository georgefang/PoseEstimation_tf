
"""
Predicting 3d poses from 2d joints
2d joints dataset is predicted by hg model from images of Human3.6M
3d joints dataset is 3d Positions of Human3.6M
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import utils3d
import utils2d
import linear_model

tf.app.flags.DEFINE_string("action","All", "The action to train on. 'All' means all the actions")
tf.app.flags.DEFINE_boolean("evaluateActionWise",True, "The dataset to use either h36m or heva")
tf.app.flags.DEFINE_string("data_dir",   "../3d-pose-baseline/data/h36m/", "Data directory")
tf.app.flags.DEFINE_string("config_file", "config/config_linear.cfg", "Config file of linear model parameters")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_string("load", None, "Try to load a previous trained model.")
tf.app.flags.DEFINE_boolean("save_predict", False, "save predict sample results")

FLAGS = tf.app.flags.FLAGS
params = utils2d.process_config(FLAGS.config_file)
train_dir = params['train_dir']
use_root = params['use_root']
if not FLAGS.sample:
  os.system('mkdir -p {}'.format(train_dir))
  os.system('cp {0} {1}'.format(FLAGS.config_file, train_dir))

def create_model( session ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """
  summaries_dir = os.path.join( train_dir, 'log' ) # Directory for TB summaries
  os.system('mkdir -p {}'.format(summaries_dir))
  model = linear_model.LinearModel(
  # model = conv_model.ConvModel(
      params['linear_size'],
      params['num_layers'],
      params['residual'],
      params['batch_norm'],
      params['max_norm'],
      params['batch_size'],
      params['learning_rate'],
      summaries_dir,
      use_root,
      dtype=tf.float32)

  if FLAGS.load is None:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
  else:
    load = os.path.join(train_dir, FLAGS.load)
    model.saver.restore( session, load )

  return model

def train():
  """Train a linear model for 3d pose estimation by using Human3.6M dataset"""
  train_dir = params['train_dir']
  print( train_dir )
  
  os.system('mkdir -p {}'.format(train_dir))

  actions = utils3d.define_actions( FLAGS.action )

  number_of_actions = len( actions )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = utils3d.read_3d_data(actions, FLAGS.data_dir, use_root)

  # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d, train_root_2dpos, test_root_2dpos = utils3d.read_2d_data(actions, FLAGS.data_dir, use_root)

  print( "done reading and normalizing data." )

  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    model = create_model( sess )
    model.train_writer.add_graph( sess.graph )
    print("Model created")

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 100

    for _ in xrange( params['epochs'] ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs = model.get_all_batches( train_set_2d, train_set_3d, training=True )
      nbatches = len( encoder_inputs )
      print("There are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =  model.step( sess, enc_in, dec_out, params['dropout'], isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batches
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===
      isTraining = False

      if FLAGS.evaluateActionWise:

        print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs

        cum_err = 0
        for action in actions:

          print("{0:<12} ".format(action), end="")
          # Get 2d and 3d testing data for this action
          action_test_set_2d = get_action_subset( test_set_2d, action )
          action_test_set_3d = get_action_subset( test_set_3d, action )
          encoder_inputs, decoder_outputs = model.get_all_batches( action_test_set_2d, action_test_set_3d, training=False)

          act_err, _, step_time, loss = evaluate_batches( sess, model,
            data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
            data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
            current_step, encoder_inputs, decoder_outputs )
          cum_err = cum_err + act_err

          print("{0:>6.2f}".format(act_err))

        summaries = sess.run( model.err_mm_summary, {model.err_mm: float(cum_err/float(len(actions)))} )
        model.test_writer.add_summary( summaries, current_step )
        print("{0:<12} {1:>6.2f}".format("Average", cum_err/float(len(actions) )))
        print("{0:=^19}".format(''))

      else:

        n_joints = 17 if use_root else 16
        encoder_inputs, decoder_outputs = model.get_all_batches( test_set_2d, test_set_3d, training=False)

        total_err, joint_err, step_time, loss = evaluate_batches( sess, model,
          data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
          data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
          current_step, encoder_inputs, decoder_outputs, current_epoch )

        print("=============================\n"
              "Step-time (ms):      %.4f\n"
              "Val loss avg:        %.4f\n"
              "Val error avg (mm):  %.2f\n"
              "=============================" % ( 1000*step_time, loss, total_err ))

        for i in range(n_joints):
          # 6 spaces, right-aligned, 5 decimal places
          print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
        print("=============================")

        # Log the error to tensorboard
        summaries = sess.run( model.err_mm_summary, {model.err_mm: total_err} )
        model.test_writer.add_summary( summaries, current_step )

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(train_dir, 'linear_model_{}'.format(current_epoch)))
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()


def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action}


def evaluate_batches( sess, model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = 17 if use_root else 16
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    # denormalize
    enc_in  = utils3d.unNormalizeData( enc_in,  data_mean_2d, data_std_2d)
    dec_out = utils3d.unNormalizeData( dec_out, data_mean_3d, data_std_3d )
    poses3d = utils3d.unNormalizeData( poses3d, data_mean_3d, data_std_3d )

    # Keep only the relevant dimensions
    # dtu3d = np.copy(dim_to_use_3d)
    jvlen = int(poses3d.shape[1]/3)


    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], jvlen) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, jvlen*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time, loss


def sample():
  """Get samples from a model and visualize them"""

  actions = utils3d.define_actions( FLAGS.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = utils3d.read_3d_data(actions, FLAGS.data_dir, use_root)

  # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d, train_root_2dpos, test_root_2dpos = utils3d.read_2d_data(actions, FLAGS.data_dir, use_root)

  print( "done reading and normalizing data." )

  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    batch_size = params['batch_size']
    model = create_model(sess)
    print("Model loaded")

    set_2d = test_set_2d
    set_3d = test_set_3d
    root_3d = test_root_positions
    root_2d = test_root_2dpos

    for key2d in set_2d.keys():

      (subj, b, fname) = key2d
      print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

      # keys should be the same if 3d is in camera coordinates
      key3d = key2d

      enc_in  = set_2d[ key2d ]
      n2d, _ = enc_in.shape
      dec_out = set_3d[ key3d ]
      n3d, _ = dec_out.shape
      enc_in_rp = root_2d[ key2d ]
      dec_out_rp = root_3d[ key3d ]

      assert n2d == n3d

      # Split into about-same-size batches
      batch_size = n2d
      enc_in   = np.array_split( enc_in,  n2d // batch_size )
      dec_out  = np.array_split( dec_out, n3d // batch_size )
      # dec_out_rp = np.array_split( dec_out_rp, n3d // batch_size )
      all_poses_3d = []

      for bidx in range( len(enc_in) ):

        # Dropout probability 0 (keep probability 1) for sampling
        dp = 1.0
        _, _, poses3d = model.step(sess, enc_in[bidx], dec_out[bidx], dp, isTraining=False)

        # denormalize
        enc_in[bidx]  = utils3d.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d )
        dec_out[bidx] = utils3d.unNormalizeData( dec_out[bidx], data_mean_3d, data_std_3d )
        poses3d       = utils3d.unNormalizeData(       poses3d, data_mean_3d, data_std_3d )
        all_poses_3d.append( poses3d )

      # Put all the poses together
      enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, all_poses_3d] )

      if not use_root:
        enc_in = np.hstack((enc_in_rp, enc_in))
        dec_out = np.hstack((dec_out_rp, dec_out))
        poses3d = np.hstack((dec_out_rp, poses3d))

      enc_in[:, 2:] = enc_in[:, 2:] + np.tile(enc_in[:, :2], [1, int(enc_in.shape[1]/2)-1])
      dec_out[:, 3:] = dec_out[:, 3:] + np.tile(dec_out[:, :3], [1, int(dec_out.shape[1]/3)-1])
      poses3d[:, 3:] = poses3d[:, 3:] + np.tile(poses3d[:, :3], [1, int(poses3d.shape[1]/3)-1])

      dec_out[:, 1::3] = -dec_out[:, 1::3]
      dec_out[:, 2::3] = -dec_out[:, 2::3]
      poses3d[:, 1::3] = -poses3d[:, 1::3]
      poses3d[:, 2::3] = -poses3d[:, 2::3]

      if FLAGS.save_predict:
        output = os.path.join(train_dir, 'output')
        os.system('mkdir -p {}'.format(output))
        file = os.path.join(output, 'S{}'.format(subj))
        if not os.path.exists(file):
          os.mkdir(file)
        storename = os.path.join(file, fname)
        storename = storename[:-3] + '.xml'
        utils3d.save_pose_data_xml(poses3d, storename, filter=4)

  # Grab a random batch to visualize
  enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, poses3d] )
  posetemp = np.copy(dec_out[:, 2::3])
  dec_out[:, 2::3] = dec_out[:, 1::3]
  dec_out[:, 1::3] = -posetemp
  posetemp = np.copy(poses3d[:, 2::3])
  poses3d[:, 2::3] = poses3d[:, 1::3]
  poses3d[:, 1::3] = -posetemp
  idx = np.random.permutation( enc_in.shape[0] )
  enc_in, dec_out, poses3d = enc_in[idx, :], dec_out[idx, :], poses3d[idx, :]

  # Visualize random samples
  import matplotlib.gridspec as gridspec

  # 1080p	= 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )

  gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx, exidx = 1, 1
  nsamples = 15
  for i in np.arange( nsamples ):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    p2d = enc_in[exidx,:]
    utils3d.show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d gt
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = dec_out[exidx,:]
    utils3d.show3Dpose( p3d, ax2 )

    # Plot 3d predictions
    ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
    p3d = poses3d[exidx,:]
    utils3d.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

    exidx = exidx + 1
    subplot_idx = subplot_idx + 3

  plt.show()


def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
