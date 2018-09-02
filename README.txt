####################################################
2D & 3D Human Pose Estimation
AvatarWorks Lab
Huanshi Ltd.

@author: Xiao-Zhi Fang
@mail: george.fang@avatarworks.com

Worked based on:
1. Stacked Hourglass Network for Human Pose Estimation
2. A simple yet effective baseline for 3d human pose estimation 
####################################################

################# 2D Part ##########################
I. CONFIG FILE
	A 'config_mpii.cfg' is present in the directory.
	It contains all the variables needed to tweak the model.
	
	training_txt_file : Path to TEXT file containing information about images
	img_directory : Path to MPII dataset
	img_size : Size of input Image /!\ DO NOT CHANGE THIS PARAMETER (256 default value)
	hm_size : Size of output heatMap /!\ DO NOT CHANGE THIS PARAMETER (64 default value)
	num_joints : Number of joints considered
	joint_list: List of joint name in MPII
	name : Name of trained model
	nFeats: Number of Features/Channels in the convolution layers (256 / 512 are good but you can set whatever you need )
	multi: Multiple of Features/Channels in Depthwise Convolutional Layer of Residual Module(only use in mobile condition)
	nStacks: Number of Stacks (4 to make faster predictions, 8 stacks are used in the paper)
	nModules : NOT USED
	nLow : Number of downsampling in one stack (default: 4 => dim 64->4)
	dropout_rate : Percentage of neurons deactivated at the end of Hourglass Module (Not Used)
	batch_size : Size of training batch (8/16/32 are good values depending on your hardware)
	nEpochs : Number of training epochs
	epoch_size : Iteration in a single epoch
	learning_rate: Starting Learning Rate
	learning_rate_decay: Decay applied to learning rate (in (0,1], 0 not included), set to 1 if you don't want decay learning rate. (Usually, keep decay between 0.9 and 0.99)
	decay_step : Step to apply decay to learning rate
	valid_iteration : Number of prediction made on validation set after one epoch (valid_iteration >= 1)
	log_dir_test : Directory to Test Log file
	log_dir_train : Directory to Train Log file
	saver_step : Step to write in train log files (saver_step < epoch_size)
	saver_directory: Directory to save trained Model

II. DATASET
	To create a dataset you need to put every images of your set on the 'img_directory'.
	Add information about your images into the 'training_txt_file':
	
	EXAMPLE:
		015601864.jpgA 291 1 896 559 594 257 3.021046 620 394 616 269 573 185 647 188 661 221 656 231 610 187 647 176 637 190 696 108 606 217 553 161 601 167 692 185 693 240 688 313
		015601864.jpgB 704 1 1199 469 952 222 2.472117 895 293 910 279 945 223 1012 218 961 315 960 403 979 221 906 190 912 191 831 182 871 304 883 229 888 174 924 206 1013 203 955 263
		015599452.jpgA 54 1 1183 720 619 329 5.641276 -1 -1 -1 -1 806 543 720 593 -1 -1 -1 -1 763 568 683 290 682 256 676 68 563 296 555 410 647 281 719 299 711 516 545 466
		015599452.jpgB 402 1 1280 720 1010 412 6.071051 -1 -1 -1 -1 987 607 1194 571 -1 -1 -1 -1 1091 589 1038 292 1025 261 947 74 914 539 955 470 931 315 1145 269 1226 475 1096 433
		
	In this example we consider 16 joints
	['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
	The text file is formalized as follow:
		image_name[LETTER] x_box_min y_box_min x_box_max y_box_max x_center y_center scale x1 y1 x2 y2 x3 y3 ...
		image_name is the file name
		[LETTER] Indicates the person considered
		(x_box_min y_box_min x_box_max y_box_max) Is the bounding box of the considered person in the scene(Not Used)
		(x_center y_center) Is the center of cropped image
		(scale) Is the size of cropped image related to 100  
		(x1 y1 x2 y2 x3 y3 ...) is the list of coordinates of every joints
	This data formalism consider a maximum of 10 persons in a single image (You can tweak the datagen.py file to consider more persons)
	
	/!\
	Missing part or values must be marked as -1

III. TRAINING
	To train a model, make sure to have a 'config_mpii.cfg' file in your main directory and a text file with regard to your dataset. Then run train_hg.py
	It will run the training.
	On a TITAN GTX for mini_batches of 16 images on 100 epochs of 1000 iterations: 2 days of training (1.6 million images)
	Training Parameters:
	'configfile': name of config file which stated in Part I
	'loadmodel': trained model if you want to continue training
	
	
IV. SAVING AND RESTORING MODELS
	Saving is automatically done when training.
	In the 'saver_directory' you will find several files:
	'name'_'epoch'.data-00000-of-00001
	'name'_'epoch'.index
	'name'_'epoch'.meta
	
	You can manually load the graph from *.meta file using TensorFlow methods. (http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)
	Or you can use the Restore Method in hourglass.py
	To do so, you first need to create the same graph as the saved one. To do so use the exact same 'config_mpii.cfg' that you used to train your model.
	Then use HourglassModel('config_mpii.cfg').restore(modelToLoad) to restore pretrained model.
	modelToLoad: 'saver_directory'/'name'_'epoch'
	/!\ BE SURE TO USE THE SAME CONFIG.CFG FILE OR THE METHOD WON'T BE ABLE TO ASSIGN THE RIGHT WEIGHTS

################ 3D Part ######################
I. CONFIG FILE
	A 'config_linear.cfg' is present in the directory.
	It contains all the variables needed to tweak the model

	linear_size: size of linear layer
	num_layers: number of linear module
	residual: whether to use redidual method
	batch_norm: whether to use batch normalization
	max_norm: whether to use max normalization
	batch_size: size of training batch (default to 64)
	learning_rate: learning rate in training linear model
	dropout: drop out rate during training 
	epochs: number of training epochs
	train_dir: directory to save the trained models
	summaries_dir: directory to save the summaries data
	use_root: whether to predict position of root joint
	model_3d: linear model name used in prediction
	mean_std_3d: name of data file of mean and standard calculated in Human3.6M dataset

II. TRAINING & SAMPLING
	To train a model, make sure to have a 'config_linear.cfg' file in your main directory and a text file with regard to your dataset. Then run train_3d.py
	Training Parameters:
	'action': choose one action or all of h3.6m to train
	'evaluateActionWise': whether to evaluate each action
	'data_dir': Directory of Human3.6M dataset 
	'config_file': Name of config file stated in Part I
	'sample': Set to True for sampling test sets of Human3.6M
	'use_cpu': whether to use cpu
	'load': Name of trained model in 'train_dir'(use in 'sample')
	'save_predict': whether to save the sample result as .xml files

	Notice: After training, you should copy the meanstd_dataset.h5(in example/models3d/baseline-3d) to your trained directory.

################# Prediction ####################
	2D and 3D prediction are both implemented in 'predict_hg3d.py', which contains some predict methods such as image, video, camera, h3.6m dataset and so on (only one method each run time). You can extend to other methods if you want.
	Prediction Parameters:
	'model_dir': directory of 2D trained model
	'config_file': name of 2D config file in 'model_dir'
	'model_file": file name of 2D trained model'
	'resize': whether to resize the input image to 256*256
	'hm': whether to show the predicted heat map
	'image_file': set image name to predict 2D pose from some image 
	'camera': set web camera index to predict 2D pose from some camera
	'video': set video name to predict 2D pose from some video
	'video_save': set save name to store the result of video prediction
	'h36mDir': set directory of h36m to predict 2D pose from h36m dataset
	'predict3d': set True to predict 3D pose
	'model3d_dir': directory of 3D trained model
	'config_file_3d': name of 3D config file in 'model3d_dir'  

#################################################
2D Pose预测运行范例：
python predict_hg3d.py --h36mDir [h36m dataset directory]
或者
python predict_hg3d.py --image_file image/yifan_1.jpg
工程给出了默认的已训练好的模型，在trained/stack2_hg/hg_model_100.
第一个例子运行的是h36m数据集，预测的输入是剪裁后的图片，自动遍历从S1-S11所有的图片。
第二个例子运行的是单张图片的Pose预测，输入的图片最好是人体完整，占比较大且尽量居中。
另外还提供了视频的预测等功能，默认方法是用OpenCV打开视频，如果OpenCV打不开视频，还提供了用imageio打开视频的方法，需安装imageio库。
如果用imageio打开视频，需将predict_hg3d.py里predict.predict_video那一行注释，去掉predict.predict_video_imageio的注释。

2D Pose训练运行范例：
python train_hg.py
训练时，需要在配置文件.cfg（默认是config/config_mpii.cfg）里修改mpii图片数据集所在的文件夹路径：img_directory
同事也要修改训练模型存放的目录：save_directory
192.168.11.200服务器上的mpii图片数据集目录为：
/home/research/disk1/george/StackedHourglass/data/mpii/images
标注信息文件默认是用infors/dataset_mpii.txt

3D Pose预测运行范例：
python predict_hg3d.py --image_file image/yifan_1.jpg --predict3d
或者
python predict_hg3d.py --video video/me_test.mp4 --predict3d --show3d
第一个例子运行是单张图片的2D和3D Pose的预测
第二个例子运行的是视频的2D和3D Pose的预测
工程给出了默认的已训练好的模型，在trained/baseline_nr/linear_model_50.
3D的Pose是用matplotlib.pyplot显示结果，显示结果比较耗时间，实际预测的速度没那么慢。如果不想看3D效果，去掉--show3d
第二个例子最后还会生成3D Pose的xml文件（用于3D动画生成的功能），在存放video的文件夹下

3D Pose训练运行范例：
python train_3d.py --data_dir [h36m dataset directory]
data_dir指的是h36m数据集存放的地址，192.168.11.200服务器上存放的路径是：
/home/research/disk1/george/3d-pose-baseline/data/h36m/
训练时，需要在配置文件.cfg（默认是config/config_linear.cfg）里修改训练模型存放的路径train_dir
这个配置文件会自动复制到train_dir里面，训练完成后，把train_dir里面的这份配置文件的model_3d改成最终的模型名称
比如迭代训练50代，那么最后的模型名称就是linear_model_50
把example/models3d/baseline-3d里面的meanstd_dataset.h5拷贝到train_dir里

3D Pose训练模型在h36m测试数据集上效果：
python train_3d.py --data_dir [h36m dataset directory] --sample --config_file [cfg file in train_dir]
如果想保存测试数据集的xml文件（用于3D动画生成的功能），在命令后面加上--save_predict，结果保存在train_dir里的output文件夹下

其他配置参数参看该文件的英文部分




