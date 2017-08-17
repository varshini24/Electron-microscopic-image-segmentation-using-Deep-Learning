"""
Date: 06/20/2017
author: varshini guddanti

subject: code to convert tiff to png format of EM neuron images (AND) histogram equalization using CLAHE.



"""

import tensorflow
import cv2
from cv2 import *
import numpy as np
import mahotas as mh
#import Image
import scipy
from scipy import ndimage
from matplotlib import pyplot as plt
import skimage
from skimage import exposure
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

train_image_path_ac3 = '/projects/visualization/turam/em-brain/data/cell_paper/ac3/ac3Data'
train_image_path_ac4 = '/projects/visualization/turam/em-brain/data/cell_paper/ac4/ac4Data'

save_path = '/home/guddanti/dev/rhoana/png_images/input'

def equalize(image):
	image = mh.imread(image)
	print image.shape

	image = exposure.equalize_adapthist(image)
	image = skimage.img_as_ubyte(image)
	print image.dtype

	return image

onlyfiles = [f for f in listdir(train_image_path_ac4) if isfile(join(train_image_path_ac4, f))]

count = 0

for img in onlyfiles:
	if str(img).__contains__('.tiff'):
		try:
			#print "check1"
			final_img = equalize(join(train_image_path_ac4, img))
			#print "check2"
			#final_img = np.reshape(final_img, (final_img.shape[0], final_img.shape[1],1))
			#print final_img.shape
			#print "check3"
			# changing image extension from tiff to png
			#img = str(img).split('.')[0]+'.png'  #images are converted from 32-bit to 16-bit when changed to png
			img = str(img).split('.')[0]+'_ac4'+'.png'

		except IOError:
			print('Unrecognized image name format!')

		count = count +1 

		
		print 'No. of images processed:', count, img

		
		
		mh.imsave(os.path.join(save_path, img), final_img)
		



