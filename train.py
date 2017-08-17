


"""
train.py
U-Net for automatic EM image segmentation
@author: Varshini Guddanti

USAGE: python train.py <tracking_int> <learning_rate_float> <num_epocs_int> 
"""




import tensorflow as tf
import cv2
import os
from os import listdir
from PIL import Image
from os.path import isfile, join
import numpy as np
import tqdm
import h5py
import deepdish as dd
from data_generation.trainingDatasetAligner import train_list_of_lists, readImagesFromPaths
import mahotas as mh
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from data_generation.data_helpers import gen_batch


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#output_file_name = 'lr_0.002_output_'+str(sys.argv[1])+'.txt'

#sys.stdout = open(output_file_name, 'w')

tf.set_random_seed(1)

#input and output images

real_test_image = '/projects/visualization/turam/em-brain/data/kasthuri11/png/kasthuri11_image_0_10752_0_13312_10_1.png'

log_folder = '/home/guddanti/dev/rhoana/log_folder'
train_dir = '/home/guddanti/dev/rhoana/checkpoint_folder/'

input_data = '/home/guddanti/dev/rhoana/png_images/input/'
target_data = '/home/guddanti/dev/rhoana/png_images/target/'

input_files = os.listdir(input_data)
target_files = os.listdir(target_data)

# hyper parameters

batch_size = 1
train_max_steps = 250
test_max_steps = 106
height = 1024
width = 1024
channels = 1

#data reading from disk
# get training data from "trainingDatasetAligner.py"
# convert numpy array into pandas dataframe
image_list = train_list_of_lists(input_files) #get list of lists containing [image, segmentation] pairs 
image_array = np.array(image_list) # convert lists into numpy array #image path array









# Constants describing the training process.

INITIAL_LEARNING_RATE = float(sys.argv[2])      # Initial learning rate.


lr = tf.placeholder(tf.float32, name = 'lr')

global_step = tf.Variable(0, trainable=False)


# learning rate decay
lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
	train_max_steps, 0.9, staircase=True)
tf.summary.scalar('learning_rate', lr)


#lr = 0.002

is_training_placeholder = tf.placeholder(tf.bool)

image_filename_placeholder = tf.placeholder(tf.string)
annotation_filename_placeholder = tf.placeholder(tf.string)
is_training_placeholder = tf.placeholder(tf.bool)





image_tensor = tf.read_file(image_filename_placeholder)
annotation_tensor = tf.read_file(annotation_filename_placeholder)


image_tensor = tf.image.decode_image(image_tensor, channels = 1)
image_tensor = tf.image.per_image_standardization(image_tensor)


annotation_tensor = tf.image.decode_jpeg(annotation_tensor, channels = 1)
annotation_tensor = tf.cast(annotation_tensor, tf.int64)/255
#annotation_tensor = tf.image.resize_images(annotation_tensor, [572, 572, 1])
#annotation_tensor = tf.reshape(annotation_tensor, [-1, 1024, 1024, 1])

#crop_fraction = (836/1024)
#annotation_tensor = tf.image.central_crop(annotation_tensor, 0.82)
annotation_tensor = tf.image.resize_image_with_crop_or_pad(annotation_tensor, 836, 836)
print "annotation_tensor: " + str(annotation_tensor)
#sess = tf.Session()
#print('annotation_image: ', sess.run(annotation_tensor))
	



# Get ones for each class instead of a number -- we need that
# for cross-entropy loss later on. Sometimes the groundtruth
# masks have values other than 1 and 0. 
class_labels_tensor = tf.equal(annotation_tensor, 1)
background_labels_tensor = tf.not_equal(annotation_tensor, 1)

# Convert the boolean values into floats -- so that
# computations in cross-entropy loss is correct
bit_mask_class = tf.to_float(class_labels_tensor)
bit_mask_background = tf.to_float(background_labels_tensor)

combined_mask = tf.concat(values=[bit_mask_class, bit_mask_background], axis = 1)

# Lets reshape our input so that it becomes suitable for 
# tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2), name = 'flat_labels')

print "flat_labels:" + str(flat_labels)

# Convert image to float32 before subtracting the
# mean pixel value
image_float = tf.to_float(image_tensor, name='ToFloat')
image_float = tf.reshape(image_float, [-1, 1024, 1024, 1])


def inference():

	#  convolutional layers with their channel counts

	K = 64  # convolutional layer output depth
	L = 128  
	M = 256  
	N = 512
	P = 1024

	# weights and biases for the U-Net layers
	W1 = tf.Variable(tf.truncated_normal([3, 3, 1, K], stddev=0.1))  # conv 3x3, RELU
	B1 = tf.Variable(tf.ones([K])/(K*K))

	W2 = tf.Variable(tf.truncated_normal([3, 3, K, K], stddev=0.1)) # conv 3x3, RELU
	B2 = tf.Variable(tf.ones([K])/(K*K))

	W2_ = tf.Variable(tf.truncated_normal([3, 3, K, K], stddev=0.1)) # conv 3x3, RELU
	B2_ = tf.Variable(tf.ones([K])/(K*K))

	W3 = tf.Variable(tf.truncated_normal([2, 2, K, K], stddev=0.1)) # deconv 3x3, RELU
	B3 = tf.Variable(tf.ones([K])/(K*K))

	W4 = tf.Variable(tf.truncated_normal([3, 3, K, L], stddev=0.1)) # conv 3x3, RELU
	B4 = tf.Variable(tf.ones([L])/(L*L))

	W5 = tf.Variable(tf.truncated_normal([3, 3, L, L], stddev=0.1))  # conv 3x3, RELU
	B5 = tf.Variable(tf.ones([L])/(L*L))

	W5_ = tf.Variable(tf.truncated_normal([3, 3, L, L], stddev=0.1))  # conv 3x3, RELU
	B5_ = tf.Variable(tf.ones([L])/(L*L))

	W6 = tf.Variable(tf.truncated_normal([2, 2, L, L], stddev=0.1))  # maxpooling(2x2)
	B6 = tf.Variable(tf.ones([L])/(L*L))

	W7 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1))  # conv 3x3, RELU
	B7 = tf.Variable(tf.ones([M])/(M*M))

	W8 = tf.Variable(tf.truncated_normal([3, 3, M, M], stddev=0.1))  # 3conv 3x3, RELU
	B8 = tf.Variable(tf.ones([M])/(M*M))

	W8_ = tf.Variable(tf.truncated_normal([3, 3, M, M], stddev=0.1))  # 3conv 3x3, RELU
	B8_ = tf.Variable(tf.ones([M])/(M*M))

	W9 = tf.Variable(tf.truncated_normal([2, 2, M, M], stddev=0.1))  # maxpooling(2x2)
	B9 = tf.Variable(tf.ones([M])/(M*M))

	W10 = tf.Variable(tf.truncated_normal([2, 2, M, N], stddev=0.1))  # maxpooling(2x2)
	B10 = tf.Variable(tf.ones([N])/(N*N))

	W11 = tf.Variable(tf.truncated_normal([3, 3, N, N], stddev=0.1))  # conv 3x3, RELU
	B11 = tf.Variable(tf.ones([N])/(N*N))

	W11_ = tf.Variable(tf.truncated_normal([3, 3, N, N], stddev=0.1))  # conv 3x3, RELU
	B11_ = tf.Variable(tf.ones([N])/(N*N))

	W12 = tf.Variable(tf.truncated_normal([2, 2, N, N], stddev=0.1))  # maxpooling(2x2)
	B12 = tf.Variable(tf.ones([N])/(N*N))

	W13 = tf.Variable(tf.truncated_normal([3, 3, N, P], stddev=0.1))  # conv 3x3, RELU
	B13 = tf.Variable(tf.ones([P])/(P*P))

	W14 = tf.Variable(tf.truncated_normal([3, 3, P, P], stddev=0.1))  # conv 3x3, RELU
	B14 = tf.Variable(tf.ones([P])/(P*P))

	W14_ = tf.Variable(tf.truncated_normal([3, 3, P, P], stddev=0.1))  # conv 3x3, RELU
	B14_ = tf.Variable(tf.ones([P])/(P*P))

	W15 = tf.Variable(tf.truncated_normal([2, 2, P, P], stddev=0.1))  # up-conv (2x2)
	B15 = tf.Variable(tf.ones([P])/(P*P))

	W16 = tf.Variable(tf.truncated_normal([3, 3, 1536, N], stddev=0.1))  # conv 3x3, RELU
	B16 = tf.Variable(tf.ones([N])/(N*N))

	W17 = tf.Variable(tf.truncated_normal([3, 3, N, N], stddev=0.1))  # conv 3x3, RELU
	B17 = tf.Variable(tf.ones([N])/(N*N))

	W17_ = tf.Variable(tf.truncated_normal([3, 3, N, N], stddev=0.1))  # conv 3x3, RELU
	B17_ = tf.Variable(tf.ones([N])/(N*N))

	W18 = tf.Variable(tf.truncated_normal([2, 2, N, N], stddev=0.1))  # up-conv (2x2)
	B18 = tf.Variable(tf.ones([N])/(N*N))

	W19 = tf.Variable(tf.truncated_normal([3, 3, 768, M], stddev=0.1))  # conv 3x3, RELU
	B19 = tf.Variable(tf.ones([M])/(M*M))

	W20 = tf.Variable(tf.truncated_normal([3, 3, M, M], stddev=0.1))  # conv 3x3, RELU
	B20 = tf.Variable(tf.ones([M])/(M*M))

	W20_ = tf.Variable(tf.truncated_normal([3, 3, M, M], stddev=0.1))  # conv 3x3, RELU
	B20_ = tf.Variable(tf.ones([M])/(M*M))

	W21 = tf.Variable(tf.truncated_normal([2, 2, M, M], stddev=0.1))  # up-conv (2x2)
	B21 = tf.Variable(tf.ones([M])/(M*M))

	W22 = tf.Variable(tf.truncated_normal([3, 3, 384, L], stddev=0.1))  # conv 3x3, RELU
	B22 = tf.Variable(tf.ones([L])/(L*L))

	W23 = tf.Variable(tf.truncated_normal([3, 3, L, L], stddev=0.1))  # conv 3x3, RELU
	B23 = tf.Variable(tf.ones([L])/(L*L))

	W23_ = tf.Variable(tf.truncated_normal([3, 3, L, L], stddev=0.1))  # conv 3x3, RELU
	B23_ = tf.Variable(tf.ones([L])/(L*L))

	W24 = tf.Variable(tf.truncated_normal([2, 2, L, L], stddev=0.1))  # up-conv (2x2)
	B24 = tf.Variable(tf.ones([L])/(L*L))

	W25 = tf.Variable(tf.truncated_normal([3, 3, 192, K], stddev=0.1))  # conv 3x3, RELU
	B25 = tf.Variable(tf.ones([K])/(K*K))

	W26 = tf.Variable(tf.truncated_normal([3, 3, K, K], stddev=0.1))  # conv 3x3, RELU
	B26 = tf.Variable(tf.ones([K])/(K*K))

	W26_ = tf.Variable(tf.truncated_normal([3, 3, K, K], stddev=0.1))  # conv 3x3, RELU
	B26_ = tf.Variable(tf.ones([K])/(K*K))

	W27 = tf.Variable(tf.truncated_normal([1, 1, K, 2], stddev=0.1))  # conv (1x1)
	B27 = tf.Variable(tf.ones([2])/(2*2))

	#layers of U-Net model
	# Unet-1 (down)
	stride = 1  # output is 1020x1020
	Y1 = tf.nn.relu(tf.nn.conv2d(image_float, W1, strides=[1, stride, stride, 1], padding= 'VALID') + B1) #padding = 'VALID' => no padding
	print "Y1:" + str(Y1)
	stride = 1  # output is 1018x1018
	Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2)
	print "Y2:" + str(Y2)
	Y2_ = tf.nn.relu(tf.nn.conv2d(Y2, W2_, strides=[1, stride, stride, 1], padding='SAME') + B2_)
	print "Y2_:" + str(Y2_)

	#Unet-2 (down)
	stride = 2
	k = 2  # output is 510x510
	Y3 = (tf.nn.max_pool(Y2_, ksize = [1,k,k,1], strides=[1, stride, stride, 1], padding='VALID'))
	print "Y3:" + str(Y3)
	stride = 1  # output is 508x508
	Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4)
	print "Y4:" + str(Y4)
	stride = 1  # output is 508x508
	Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5)
	print "Y5:" + str(Y5)
	Y5_ = tf.nn.relu(tf.nn.conv2d(Y5, W5_, strides=[1, stride, stride, 1], padding='SAME') + B5_)
	print "Y5_:" + str(Y5_)

	#Unet-3 (down)
	stride = 2  # output is 254x254
	K = 2
	Y6 = (tf.nn.max_pool(Y5, ksize = [1,k,k,1], strides=[1, stride, stride, 1], padding='VALID'))
	print "Y6:" + str(Y6)
	stride = 1  # output is 252x252
	Y7 = tf.nn.relu(tf.nn.conv2d(Y6, W7, strides=[1, stride, stride, 1], padding='VALID') + B7)
	print "Y7:" + str(Y7)
	stride = 1  # output is 250x250
	Y8 = tf.nn.relu(tf.nn.conv2d(Y7, W8, strides=[1, stride, stride, 1], padding='VALID') + B8)
	print "Y8:" + str(Y8)
	Y8_ = tf.nn.relu(tf.nn.conv2d(Y8, W8_, strides=[1, stride, stride, 1], padding='SAME') + B8_)
	print "Y8_:" + str(Y8_)

	#Unet-4(down)
	stride = 2  # output is 
	K = 2
	Y9 = (tf.nn.max_pool(Y8, ksize = [1,k,k,1], strides=[1, stride, stride, 1], padding='VALID'))
	print "Y9:" + str(Y9) 
	stride = 1  # output is 
	Y10 = tf.nn.relu(tf.nn.conv2d(Y9, W10, strides=[1, stride, stride, 1], padding='VALID') + B10)
	print "Y10:" + str(Y10)
	stride = 1  # output is 
	Y11 = tf.nn.relu(tf.nn.conv2d(Y10, W11, strides=[1, stride, stride, 1], padding='VALID') + B11)
	print "Y11:" + str(Y11)
	Y11_ = tf.nn.relu(tf.nn.conv2d(Y11, W11_, strides=[1, stride, stride, 1], padding='SAME') + B11_)
	print "Y11_:" + str(Y11_)
	

	#Unet-5(lowest)
	stride = 2  # output is 
	K = 2
	Y12 = (tf.nn.max_pool(Y11, ksize = [1,k,k,1], strides=[1, stride, stride, 1], padding='VALID') )
	print "Y12:" + str(Y12)
	stride = 1  # output is 
	Y13 = tf.nn.relu(tf.nn.conv2d(Y12, W13, strides=[1, stride, stride, 1], padding='VALID') + B13)
	print "Y13:" + str(Y13)
	stride = 1  # output is 
	Y14 = tf.nn.relu(tf.nn.conv2d(Y13, W14, strides=[1, stride, stride, 1], padding='VALID') + B14)
	print "Y14:" + str(Y14)
	Y14_ = tf.nn.relu(tf.nn.conv2d(Y14, W14_, strides=[1, stride, stride, 1], padding='SAME') + B14_)
	print "Y14_:" + str(Y14_)
	
	
	
	#Unet-6(up)

	Y11_crop = tf.image.resize_image_with_crop_or_pad(Y11, 112, 112)
	print "Y11_crop:" + str(Y11_crop)

	stride = 2  # output is 

	Y15 = (tf.nn.conv2d_transpose(Y14, W15, output_shape = [batch_size, 112, 112, 1024], strides=[1, stride, stride, 1], padding='SAME', name = 'Y15') + B15)
	print "Y15_:"+str(Y15)
	# crop Y11 and concatenate with Y15
	#Y11_crop = tf.image.central_crop(Y11, (Y11.shape[1]/Y15.shape[1]))
	#Y11_crop = tf.map_fn(lambda img: tf.image.central_crop(Y11, (Y11.shape[1]/Y15.shape[1])),
	#												 Y11, parallel_iterations=8, name="cropUnet6Up")
	Y15 = tf.concat([Y11_crop, Y15], axis = 3, name = 'concatUp1')
	print "Y15:"+str(Y15)


	stride = 1  # output is
	Y16 = tf.nn.relu(tf.nn.conv2d(Y15, W16, strides=[1, stride, stride, 1], padding='VALID') + B16)
	print "Y16:" + str(Y16)
	stride = 1  # output is 
	Y17 = tf.nn.relu(tf.nn.conv2d(Y16, W17, strides=[1, stride, stride, 1], padding='VALID') + B17)
	print "Y17:" + str(Y17)
	Y17_ = tf.nn.relu(tf.nn.conv2d(Y17, W17_, strides=[1, stride, stride, 1], padding='SAME') + B17_)
	print "Y17_:" + str(Y17_)

	

	#Unet-7(up)
	Y8_crop = tf.image.resize_image_with_crop_or_pad(Y8, 216, 216)
	print "Y8_crop:" + str(Y8_crop)

	stride = 2  # output is 
	Y18 = (tf.nn.conv2d_transpose(Y17, W18, output_shape = [batch_size, 216, 216, 512], strides=[1, stride, stride, 1], padding='SAME') + B18)
	
	Y18 = tf.concat([Y8_crop, Y18], axis = 3, name = 'concatUp2')
	print "Y18:"+str(Y18)

	stride = 1  # output is 
	Y19 = tf.nn.relu(tf.nn.conv2d(Y18, W19, strides=[1, stride, stride, 1], padding='VALID') + B19)
	print "Y19:"+str(Y19)
	stride = 1  # output is 
	Y20 = tf.nn.relu(tf.nn.conv2d(Y19, W20, strides=[1, stride, stride, 1], padding='VALID') + B20)
	print "Y20:"+str(Y20)
	Y20_ = tf.nn.relu(tf.nn.conv2d(Y20, W20_, strides=[1, stride, stride, 1], padding='SAME') + B20_)
	print "Y20_:"+str(Y20_)
	

	#Unet-8(up)
	Y5_crop = tf.image.resize_image_with_crop_or_pad(Y5, 424, 424)
	print "Y5_crop:" + str(Y5_crop)

	stride = 2  # output is 
	Y21 = (tf.nn.conv2d_transpose(Y20, W21, output_shape = [batch_size, 424, 424, 256], strides=[1, stride, stride, 1], padding='SAME') + B21)
	
	Y21 = tf.concat([Y5_crop, Y21], axis = 3, name = 'concatUp3')
	print "Y21:"+str(Y21)

	stride = 1  # output is 
	Y22 = tf.nn.relu(tf.nn.conv2d(Y21, W22, strides=[1, stride, stride, 1], padding='VALID') + B22)
	print "Y22:" + str(Y22)
	stride = 1  # output is 
	Y23 = tf.nn.relu(tf.nn.conv2d(Y22, W23, strides=[1, stride, stride, 1], padding='VALID') + B23)
	print "Y23:" + str(Y23)
	Y23_ = tf.nn.relu(tf.nn.conv2d(Y23, W23_, strides=[1, stride, stride, 1], padding='SAME') + B23_)
	print "Y23_:" + str(Y23_)

	#Unet-9(up-top)
	Y2_crop = tf.image.resize_image_with_crop_or_pad(Y2, 840, 840)
	print "Y2_crop:" + str(Y2_crop)

	stride = 2 # output is 
	Y24 = (tf.nn.conv2d_transpose(Y23, W24, output_shape = [batch_size, 840, 840, 128], strides=[1, stride, stride, 1], padding='SAME') + B24)
	
	Y24 = tf.concat([Y2_crop, Y24], axis = 3, name = 'concatUp4')
	print "Y24:"+str(Y24)

	stride = 1  # output is 
	Y25 = tf.nn.relu(tf.nn.conv2d(Y24, W25, strides=[1, stride, stride, 1], padding='VALID') + B25)
	print "Y25:"+str(Y25)

	stride = 1  # output is 
	Y26 = tf.nn.relu(tf.nn.conv2d(Y25, W26, strides=[1, stride, stride, 1], padding='VALID') + B26)
	print "Y26:" + str(Y26)

	Y26_ = tf.nn.relu(tf.nn.conv2d(Y26, W26_, strides=[1, stride, stride, 1], padding='SAME') + B26_)
	print "Y26_:" + str(Y26_)

	stride = 1
	Ylogits = (tf.nn.conv2d(Y26, W27, strides=[1, stride, stride, 1], padding='VALID') + B27)
	print "Ylogits:" + str(Ylogits)

	return Ylogits
	

#tf.histogram_summary('logits', Ylogits)

Ylogits = inference()

number_of_classes = 2
flat_logits = tf.reshape(tensor=Ylogits, shape=(-1, 2), name = 'flat_logits')

#print flat_logits

# define cost/loss & optimizer
#Define cross-entropy (the mean distance between Ypred and Y_)
print "logits:"+str(flat_logits)
print "labels:"+str(flat_labels)
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits = flat_logits, labels = flat_labels)

cross_entropy_sum = tf.reduce_mean(cross_entropies)

tf.summary.scalar('loss', cross_entropy_sum)
# Tensor to get the final prediction for each pixel -- pay 
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
pred = tf.argmax(Ylogits, dimension=3)

#final labels and logits
final_flat_labels = tf.argmax(flat_labels, 1)
final_flat_logits = tf.argmax(flat_logits, 1)

# Add the Op to compare the logits to the labels during evaluation.
#conf_matrix = tf.contrib.metrics.confusion_matrix(labels = final_flat_labels, predictions = final_flat_logits)
correct_prediction = tf.equal(final_flat_logits, final_flat_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy: '+str(accuracy))

probabilities = tf.nn.softmax(Ylogits)

print "Prob:"+str(probabilities)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy_sum, global_step = global_step)


# Add summary op for the loss -- to be able to see it in
# tensorboard.
tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

# Put all summary ops into one op. Produces string when
# you run it.
merged_summary_op = tf.summary.merge_all()



# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
	os.makedirs(log_folder)

# Create the checkpoint folder for saving model if doesn't exist yet
if not os.path.exists(train_dir):
	os.makedirs(train_dir)


# initialize
init = tf.global_variables_initializer()





with tf.Session() as sess:

	# Create the summary writer -- to write all the logs
	# into a specified file. This file can be later read
	# by tensorboard.
	summary_string_writer = tf.summary.FileWriter(log_folder, sess.graph)
	
	# Generate input data batches
	zipped_data = zip(image_array[:,0], image_array[:,1])
	
	# put first 250 image pairs as training data
	train_data = zipped_data[0:250]
	
	# put next 105 image pairs as testing data
	test_data = zipped_data[250:-1]
	
	#remaining one image pair is used to view the predicted image on how the model performs on a new data given
	remain_test = zipped_data[-1]
	

	#np.save('test_data', test_data)
	
	saver = tf.train.Saver()

	sess.run(init)

	# train my model
	print('************************************')
	print('Learning started. It takes sometime.')

	acc_arr = []
	loss_arr = []
	lr_arr = []
	train_epoch_duration = []
	test_epoch_duration = []
	test_acc_arr = []
	test_loss_arr = []

	num_epochs = sys.argv[3] # 4th argument given in the console for setting number of epochs (see program description)
	
	for epoch in range(int(num_epochs)):

		train_start = time.time() # start of training time

		train_batches = gen_batch(train_data, batch_size, train_max_steps)
		i = -1
		for train_batch in train_batches:
			

			i += 1

			# Get next input train data batch

			#train_batch = next(train_batches)
			train_images_batch, train_target_batch = zip(*train_batch)

		
			feed_dict_to_use_train = {image_filename_placeholder: train_images_batch[0],
			            annotation_filename_placeholder: train_target_batch[0],
			            is_training_placeholder: True}


			_, loss = sess.run([train_step, cross_entropy_sum], feed_dict = feed_dict_to_use_train)
			acc = sess.run(accuracy, feed_dict = feed_dict_to_use_train)
			lr_rate = sess.run(lr, feed_dict = feed_dict_to_use_train)
			final_logit = sess.run(final_flat_logits, feed_dict = feed_dict_to_use_train)
			final_label = sess.run(final_flat_labels, feed_dict = feed_dict_to_use_train)

			train_confusion_matx = confusion_matrix(final_label, final_logit, labels = [1,0])

			# store loss, accuracy and Learning rate to observe trends
			loss_arr.append(float(loss))
			acc_arr.append(float(acc))
			lr_arr.append(float(lr_rate))
			
			# print model performance for every 20 iterations in an epoch
			if i % 20 == 0:

				
				print('Step %d: train loss = %.8f train acc = %.8f lr = %.8f' % (i, loss,acc, lr_rate))
				print('confusion matrix:' +str(train_confusion_matx))

				summary_string = sess.run(merged_summary_op, feed_dict = feed_dict_to_use_train)

				pred_np, probabilities_np = sess.run([pred, probabilities], feed_dict = feed_dict_to_use_train)

				summary_string_writer.add_summary(summary_string, i)
				
				summary_string_writer.flush()



			# Save a checkpoint and evaluate the model periodically.
			if (i+1) % 100 == 0 or (i + 1) == train_max_steps:
				saver.save(sess, save_path = train_dir, latest_filename = 'ckpt.2017.08.02_epoch_1_model'+str(sys.argv[1]), global_step = i)
				print('Saved checkpoint')



		loss_arr2 = np.array(loss_arr)
		acc_arr2 = np.array(acc_arr)
		lr_arr2 = np.array(lr_arr)

		np.save('train_loss'+str(sys.argv[1]), loss_arr2)
		np.save('train_acc'+str(sys.argv[1]), acc_arr2)
		np.save('lr'+str(sys.argv[1]), lr_arr2)

		print('Epoch %d: Final train loss = %.8f Final train acc = %.8f' % (epoch, np.mean(loss_arr2),np.mean(acc_arr2)))
		train_duration = time.time() - train_start # end of training time
		train_epoch_duration.append(float(train_duration))

		train_epoch_time = np.array(train_epoch_duration)
		np.save('train_epoch_time'+str(sys.argv[1]), train_epoch_time)
		print('Train Time taken at Epoch %d: (%.3f sec)' % (epoch, train_duration))
			
		#end of training
		feed_dict_to_use_train[is_training_placeholder] = False
		
		print('------------------------------------')
		
	print('End of training')
	print('------------------------------------')	
	print('Testing phase')

	test_start = time.time()#start of test time
	test_itr = -1

	test_batches = gen_batch(test_data, batch_size, test_max_steps)
	for test_batch in test_batches:
		test_itr += 1
		#get next test data batch
		#test_batch = next(test_batches)
		test_images_batch, test_target_batch = zip(*test_batch)

		test_acc = sess.run(accuracy, feed_dict={image_filename_placeholder: test_images_batch[0], annotation_filename_placeholder: test_target_batch[0]})
		test_loss = sess.run(cross_entropy_sum, feed_dict={image_filename_placeholder: test_images_batch[0], annotation_filename_placeholder: test_target_batch[0]})

		final_logit = sess.run(final_flat_logits, feed_dict = {image_filename_placeholder: test_images_batch[0], annotation_filename_placeholder: test_target_batch[0]})
		final_label = sess.run(final_flat_labels, feed_dict = {image_filename_placeholder: test_images_batch[0], annotation_filename_placeholder: test_target_batch[0]})

		test_confusion_matx = confusion_matrix(final_label, final_logit, labels = [1,0])


		#store test loss and accuracy to observe trends
		test_acc_arr.append(float(test_acc))
		test_loss_arr.append(float(test_loss))

		if test_itr %20 == 0:
			print('Step %d: test loss = %.8f test acc = %.8f' % (test_itr, test_loss, test_acc))
			print('confusion matrix:'+str(test_confusion_matx))

	test_acc_arr2 = np.array(test_acc_arr)
	test_loss_arr2 = np.array(test_loss_arr)

	np.save('test_acc'+str(sys.argv[1]), test_acc_arr2)
	np.save('test_loss'+str(sys.argv[1]), test_loss_arr2)

	print('Epoch %d: Final test loss = %.8f Final test acc = %.8f' % (epoch, np.mean(test_loss_arr2), np.mean(test_acc_arr2)))
	test_duration = time.time() - test_start
	test_epoch_duration.append(float(test_duration))

	test_epoch_time = np.array(test_epoch_duration)
	np.save('test_epoch_time'+str(sys.argv[1]), test_epoch_time)
	print('Test Time taken at Epoch %d: (%.3f sec)' % (epoch, test_duration))
	print('************************************')	

	print('Predicting image')
	print remain_test[0]
	
	predict = pred.eval({image_filename_placeholder: remain_test[0], is_training_placeholder : False}, session = sess)
	#print confusion_matx
	print predict
	predict = np.reshape(predict, (836,836))
	np.save('pred_image'+str(sys.argv[1]), predict)
	dd.io.save('/home/guddanti/dev/rhoana/src/1_epoch_lr.h5', predict)

	

	print "accuracy", sess.run(accuracy, feed_dict={image_filename_placeholder: remain_test[0], annotation_filename_placeholder: remain_test[1], is_training_placeholder : False})
	print "loss", sess.run(cross_entropy_sum, feed_dict={image_filename_placeholder: remain_test[0], annotation_filename_placeholder: remain_test[1], is_training_placeholder : False})
	
	

		
	
	print('************************************')
		


