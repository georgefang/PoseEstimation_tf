
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xml.dom import minidom
import h5py
import glob
import copy

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Thorax'
SH_NAMES[8]  = 'Neck/Nose'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

# matched joint name in aw human.skeleton
joint3dNames = ['hips1', 'rightUpLeg', 'rightLeg', 'rightFoot', 'leftUpLeg', 'leftLeg', 'leftFoot', \
      'spine1', 'neck', 'head', 'head1', 'leftArm', 'leftForearm', 'leftHand', 'rightArm', 'rightForearm', 'rightHand']

def load_data_3d( bpath, subjects, actions):
  """
  Loads 3d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    bpath: String. Path where to load the data from
    subjects: List of integers. Subjects whose data will be loaded
    actions: List of strings. The actions to load
  Returns:
    data: Dictionary with keys k=(subject, action, seqname)
      values v=(nx(32*2) matrix of 2d ground truth)
      There will be 2 entries per subject/action if loading 3d data
  """
  data = {}

  for subj in subjects:
    for action in actions:

      # print('Reading subject {0}, action {1}'.format(subj, action))

      posename = '3D_positions'
      dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoses/3D_positions_Camera', '{0}*.h5'.format(action) )
      # print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          # print( fname )
          loaded_seqs = loaded_seqs + 1

          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f[posename][:]

          poses = poses.T
          # poses transform to camera ordinary
          poses[:, 1::3] = -poses[:, 1::3]
          poses[:, 2::3] = -poses[:, 2::3]
          data[ (subj, action, seqname) ] = poses

      assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )

  return data


def load_stacked_hourglass(data_dir, subjects, actions):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Directory where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
    actions: list of strings. The actions to load.
  Returns
    data: dictionary with keys k=(subject, action, seqname)
          values v=(nx(32*2) matrix of 2d stacked hourglass detections)
          There will be 2 entries per subject/action if loading 3d data
          There will be 8 entries per subject/action if loading 2d data
  """
  # Permutation that goes from SH detections to H36M ordering.
  SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
  assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

  data = {}

  for subj in subjects:
    for action in actions:

      # print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'StackedHourglass/{0}*.h5'.format(action) )
      # print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )
        seqname = seqname.replace('_',' ')

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          # print( fname )
          loaded_seqs = loaded_seqs + 1

          # Load the poses from the .h5 file
          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['poses'][:]

            # Permute the loaded data to make it compatible with H36M
            poses = poses[:,SH_TO_GT_PERM,:]

            # Reshape into n x (32*2) matrix
            poses = np.reshape(poses,[poses.shape[0], -1])
            poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

            dim_to_use_x    = np.where(np.array([x != '' and x != 'Spine' for x in H36M_NAMES]))[0] * 2
            dim_to_use_y    = dim_to_use_x+1

            dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses
            data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
      if (subj == 11 and action == 'Directions'): # <-- this video is damaged
        assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
      else:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

  return data

def normalization_stats(complete_data, dim, use_root=False ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
    use_root. boolean. Whether to use root joint
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  # data_mean = np.mean(complete_data, axis=0)
  # data_std  =  np.std(complete_data, axis=0)

  # Encodes which 17 (or 14) 2d-3d pairs we are predicting
  dimensions_to_ignore = []
  if dim == 2:
    if use_root:
      dimensions_to_use  = np.where(np.array([x != '' and x != 'Spine' for x in H36M_NAMES]))[0]
    else:
      dimensions_to_use  = np.where(np.array([x != '' and x != 'Spine' and x != 'Hip' for x in H36M_NAMES]))[0]

    dimensions_to_use    = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*2), dimensions_to_use )
  else: # dim == 3
    dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
    if not use_root:
      dimensions_to_use = np.delete( dimensions_to_use, 0)

    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3,
                                             dimensions_to_use*3+1,
                                             dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*3), dimensions_to_use )

  data_mean = np.mean(complete_data[:, dimensions_to_use], axis=0)
  data_std  = np.std(complete_data[:, dimensions_to_use], axis=0)

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use



def normalize_data(data, data_mean, data_std, dim_to_use ):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
  data_out = {}

  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean
    stddev = data_std
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

  return data_out


def unNormalizeData(normalized_data, data_mean, data_std):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  # orig_data = np.zeros((T, D), dtype=np.float32)
  # dimensions_to_use = np.array([dim for dim in range(D)
  #                               if dim not in dimensions_to_ignore])

  # orig_data[:, dimensions_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(normalized_data, stdMat) + meanMat
  return orig_data


def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  # without Sitting and SittingDown as they are difficult to be predicted in sh project
  # actions = ["Directions","Discussion","Eating","Greeting",
  #          "Phoning","Photo","Posing","Purchases",
  #          "Smoking","Waiting",
  #          "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def read_2d_data(actions, data_dir, use_root):

  print('2d data reading...')
  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
  test_set = load_stacked_hourglass( data_dir, TEST_SUBJECTS, actions)

  train_set, train_root_positions = postprocess_data( train_set, dim=2, use_root=use_root )
  test_set,  test_root_positions  = postprocess_data( test_set, dim=2, use_root=use_root )

  complete_train = copy.deepcopy(np.vstack(train_set.values()))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=2, use_root=use_root)

  train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
  test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def read_3d_data( actions, data_dir, use_root):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    use_root: whether to train and predict root joint
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
  # Load 3d data
  print('3d data reading...')
  train_set = load_data_3d( data_dir, TRAIN_SUBJECTS, actions)
  test_set  = load_data_3d( data_dir, TEST_SUBJECTS,  actions)
  
  train_root_positions = []
  test_root_positions = []
  # Apply 3d post-processing (centering around root)
  train_set, train_root_positions = postprocess_data( train_set, dim=3, use_root=use_root )
  test_set,  test_root_positions  = postprocess_data( test_set, dim=3, use_root=use_root )

  # Compute normalization statistics
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, use_root=use_root )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def postprocess_data( poses_set, dim, use_root ):
  """
  Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = copy.deepcopy(poses_set[k][:,:dim])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[:,:dim], [1, len(H36M_NAMES)] )

    if use_root:
      poses[:, :dim] = np.copy(root_positions[k])
      poses[:,:dim] = poses[:,:dim] - np.tile( poses[0,:dim], [poses.shape[0], 1])

    poses_set[k] = poses

  return poses_set, root_positions

def read_data_input(data_dir, data_mean, data_std, dim_to_use):
  """
  load 2d input data from 2d-pose-predict demo,
  these data are not from h3.6m dataset
  """
  SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
  assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

  data = {}

  print(data_dir)
  fname = data_dir
  seqname = os.path.basename( fname )

  with h5py.File( fname, 'r' ) as h5f:
    poses = h5f['2D_positions'][:]
  # poses = poses.T
  # data[seqname] = poses
  poses = poses[:,SH_TO_GT_PERM,:]

  # Reshape into n x (32*2) matrix
  poses = np.reshape(poses,[poses.shape[0], -1])
  poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

  dim_to_use_x    = np.where(np.array([x != '' and (x != 'Spine') for x in H36M_NAMES]))[0] * 2
  dim_to_use_y    = dim_to_use_x+1

  dim_to_use_xy = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
  dim_to_use_xy[0::2] = dim_to_use_x
  dim_to_use_xy[1::2] = dim_to_use_y
  poses_final[:,dim_to_use_xy] = poses
  data[ seqname ] = poses_final
  print("data size is: {}".format(poses_final.shape))
  print("dim_to_use size: {}".format(dim_to_use.shape[0]))

  data, data_root = postprocess_data(data, dim=2)
  data = normalize_data( data, data_mean, data_std, dim_to_use)

  return data, data_root

def save_pose_data_xml(pose, name, dim=3, filter=None):
  data = np.copy(pose)
  data= np.reshape(data, (data.shape[0], -1))
  assert data.shape[1] % dim == 0
  dnum = data.shape[0]
  jnum = int(data.shape[1]/dim)

  if filter:
    datatemp = np.copy(data)
    for i in range(dnum):
      den = filter
      if i<filter:
        den = i
      elif dnum-1-i < filter:
        den = dnum-1-i
      data[i] = np.mean(datatemp[i-den:i+den+1], axis=0)

  impl = minidom.getDOMImplementation()
  doc = impl.createDocument(None, None, None)
  rootElement = doc.createElement('Pose{}d'.format(dim))
  for id in range(dnum):
    frameElement = doc.createElement('frame')
    frameElement.setAttribute('id', str(id))
    for nj in range(jnum):
      jointElement = doc.createElement('joint')
      jointElement.setAttribute('name', joint3dNames[nj])
      jointElement.setAttribute('xpos', str(data[id,nj*dim+0]))
      jointElement.setAttribute('ypos', str(data[id,nj*dim+1]))
      jointElement.setAttribute('zpos', str(data[id,nj*dim+2]))
      frameElement.appendChild(jointElement)
    rootElement.appendChild(frameElement)
  doc.appendChild(rootElement)
  f = open(name, 'w')
  doc.writexml(f, addindent='  ', newl='\n')
  f.close()

def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
  """
  Visualize a 3d skeleton

  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """
  len3d = 17
  assert channels.size == len3d*3, "channels should have 51 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len3d, -1) )

  # I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  # J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  # I   = np.array([0,1,2,0,6,7,0, 12,13,14,13,17,18,13,25,26]) # start points
  # J   = np.array([1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]) # end points
  I = np.array([0,1,2,0,4,5,0,7,8, 9, 8,11,12, 8,14,15])
  J = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
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

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  # ax.get_zaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """
  len2d = 16
  assert channels.size == len2d*2, "channels should have 32 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len2d, -1) )

  # I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  # J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  # I   = np.array([0,1,2,0,6,7,0, 13,14,13,17,18,13,25,26]) # start points
  # J   = np.array([1,2,3,6,7,8,13,14,15,17,18,19,25,26,27]) # end points
  I = np.array([0,1,2,0,4,5,0,7,8, 7,10,11, 7,13,14])
  J = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 350 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')