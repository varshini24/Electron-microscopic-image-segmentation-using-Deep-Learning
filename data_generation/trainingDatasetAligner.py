"""

author: varshini guddanti

date: 06/22/2017

Subject: take images from both folder (input_data_png and target_data_png and make a list of both of them side by side)

looks like : [input_image	target_image]
"""


import os
import numpy as np
import pandas as pd
import mahotas as mh
import os
from os import listdir
from os.path import isfile, join

input_data = '/home/guddanti/dev/rhoana/png_images/input/'
target_data = '/home/guddanti/dev/rhoana/png_images/target/'

input_files = os.listdir(input_data)
target_files = os.listdir(target_data)
#print input_files


image_list = []


def train_list_of_lists(input_files):
	for file1 in input_files:

		try:

			#deleting additional filenames formed while creation of folders and other unrecognized formats
			if(str(file1).startswith('._')):
				continue

			# ac3 data processing
			elif(str(file1).__contains__('ac3')):
				input_num = str(file1).split('_ac3')[0].split('image_')[1]

				temp_str = ''
				temp_str = 'ac3ac4_neuron_' + input_num + '.jpeg'

				image_list.append([join(input_data, file1), join(target_data, temp_str)])

			# ac4 data processing
			elif(str(file1).__contains__('ac4')):
				input_num = str(file1).split('_ac4')[0].split('image_')[1]

				temp_str = ''
				temp_str = 'ac3ac4_ac4_neuron_truth_' + input_num + '.jpeg'

				image_list.append([join(input_data, file1), join(target_data, temp_str)])

		except IndexError:
			print 'unrecognized filename accessed: ' + str(file1)


	return image_list

# uncomment these lines if you need to print the output
#train_x = train_list_of_lists(input_files)
#print train_x








def readImagesFromPaths(input_df):
	"""
	reads in the input_df with the image paths and updates it with the image content.

	Parameters: input_df - input file and target file paths
	"""

	image_content_column = []
	target_image_content_column = []
	for index, row in input_df.iterrows():

		image_content = mh.imread(row['input'], as_grey=True)
		#print image_content.shape
		#image_content.reshape(1024,1024,1)
		image_content = image_content.astype(np.float32, copy=False)
		image_content = np.reshape(image_content, (image_content.shape[0], image_content.shape[1],1))
		

		target_content = mh.imread(row['target'])
		#target_content.reshape(target_content.shape + (1,))
		target_content = target_content.astype(np.float32, copy=False)
		target_content = np.reshape(target_content, (target_content.shape[0], target_content.shape[1],1))
		


		image_content_column.append(image_content)
		target_image_content_column.append(target_content)

	input_df['input_image_content'] = image_content_column
	input_df['target_image_content'] = target_image_content_column

	return input_df
