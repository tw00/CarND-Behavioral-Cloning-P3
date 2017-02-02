#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import pickle
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split

class DataGenerator(iter):
	"""
	This class serves the following purposes:
	"""
	# orginal size: 320x160 RGB 8-bit
	IMG_W = 32*4; # = 128
	IMG_H = 16*4; # = 64
	#steering_offset = 0.25;
	steering_offset = 0.1;

	# ----------------------------------------------------------------------------------------
	def __init__(self, csv_file):
		# TODO: Allow csv_file list
		self.csv_file = csv_file;

		csv = pd.read_csv(csv_file)

		for i in csv.columns:
			if isinstance(csv[i][1], str):
				csv[i] = csv[i].map(str.strip)

		img_counter = 0;

		for i in range(len(csv)):
			left_img   = './data3/IMG/' + os.path.basename(csv['left'][i]);
			right_img  = './data3/IMG/' + os.path.basename(csv['right'][i]);
			center_img = './data3/IMG/' + os.path.basename(csv['center'][i]);
			steering_angle = csv['steering'][i];

			for camera, img_path in (('L', left_img), ('R', right_img), ('C', center_img)):
		#    for camera, img_path in [('C', center_img)]:

				if camera == 'L': steering_angle = steering_angle + steering_offset;  # TESTEN
				if camera == 'R': steering_angle = steering_angle - steering_offset;  # TESTEN
		#       EVTL: np.clip, np.min

				features[img_counter,:,:,:] = img;
				labels[img_counter] = steering_angle;
				img_counter += 1;

				features[img_counter,:,:,:] = flipped;
				labels[img_counter] = -steering_angle;
				img_counter += 1;

		self.data = csv;

	# ----------------------------------------------------------------------------------------
	def filter_data(remove_car_not_moving = True):
		if remove_car_not_moving:
			N = self.data.shape[0]
			self.data = self.data[self.data.Speed >= 10.]
			print("%d samples removed due to speed < 10" % (N - self.data.shape[0]))

	# ----------------------------------------------------------------------------------------
	def calc_weights():
		"""
		calculate higher probabilty for turns
		"""

	# ----------------------------------------------------------------------------------------
	def shuffle():

	# ----------------------------------------------------------------------------------------
	def smooth_steering():

	# ----------------------------------------------------------------------------------------
	def mirror()

	# ----------------------------------------------------------------------------------------
	def split()
		#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
		X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0)
		features = None;
#labels = None;

				img = cv2.imread(img_path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV does not use RGB, it uses BGR
				img = cv2.resize(img, (IMG_W, IMG_H))
				flipped = cv2.flip(img,1)

	# ----------------------------------------------------------------------------------------
    def __next__(self):
        self.index %= self.size()
        self.index += 1
        return self.log.iloc[self.index-1]

    def __iter__(self):
		return self


num_of_samples = len(csv) * 3 * 2; # length * 3 camera angles * 2 flipped

X_test = None; X_train = None;
y_test = None; y_train = None;
features = None; label = None;
features = np.zeros(shape=[num_of_samples,IMG_H,IMG_W,3]);
labels   = np.zeros(shape=[num_of_samples])

