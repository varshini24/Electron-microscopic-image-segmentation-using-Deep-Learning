"""
Date: 06/13/2017
author: varshini guddanti

subject: code to produce binary images of hand-anotated EM neuron images.

process: 
		 1. make foreground pixels 1s and background to 0s of ground truth images ( Now let's call this IMG)
		 2. compute a boundary map using skimage.segmentation.find_boundaries (say MAP)
		 3. search for pixels in IMG and MAP. When a pixel is 1 in both, then make it 0 in IMG

"""

# imports

import tensorflow
import cv2
from cv2 import *
import numpy as np
import mahotas
import scipy.misc
from scipy import ndimage
from matplotlib import pyplot as plt
import skimage
from skimage import segmentation
import os
from os import listdir
from os.path import isfile, join

train_image_path_ac3 = '/projects/visualization/turam/em-brain/data/cell_paper/ac3/ac3neuron'
train_image_path_ac4 = '/projects/visualization/turam/em-brain/data/cell_paper/ac4/ac4neuron'

save_path = '/home/guddanti/dev/rhoana/png_images/target'


# method to binarize images
def binarize_neurons(input_image):
	input_image = mahotas.imread(input_image)
	#input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

	boundary_img = skimage.segmentation.find_boundaries(input_image, mode = 'inner', background = 0)

	binary_neuron = np.copy(input_image)
	binary_neuron[binary_neuron > 0] = 1

	new_boundary = np.copy(boundary_img)
	new_binary_neuron = np.copy(binary_neuron)
	
	for i in range(boundary_img.shape[1]):
		for j in range(boundary_img.shape[1]):
			if(binary_neuron[i][j] == 1 and boundary_img[i][j] == 1):
				new_binary_neuron[i][j] = 0

	return new_binary_neuron

onlyfiles = [f for f in listdir(train_image_path_ac3) if isfile(join(train_image_path_ac3, f))]

count = 0

for img in onlyfiles:
	if str(img).__contains__('.tiff'):
		try:
			final_img = binarize_neurons(join(train_image_path_ac3, img))
			#final_img = np.reshape(final_img, (final_img.shape[0], final_img.shape[1],1))
			#print final_img.shape
			# changing image extension from tiff to png
			img = str(img).split('.')[0]+'.jpeg'  #images are converted from 32-bit to 16-bit when changed to png
		except IOError:
			print('Unrecognized image name format!')

		count = count +1 
		print 'No. of images processed:', count, img

		

		scipy.misc.imsave(os.path.join(save_path, img), final_img)
		#mahotas.imsave(os.path.join(save_path, img), final_img)  # commented for now

