# -*- coding: utf-8 -*-

import matplotlib
import scipy
import numpy as np
import pandas as pd
import os
import pickle
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split

class DataGenerator:#(iter):
    """
    This class serves the following purposes:
    """
    # orginal size: 320x160 RGB 8-bit
    IMG_W = 128;
    IMG_H = 64;
    steering_offset = 0.1;

    # ----------------------------------------------------------------------------------------
    def mod_identity(img, steering):
        return (img, steering)

    def mod_flip(img, steering):
        img  = cv2.flip(img,1);
        steering = -steering;
        return (img, steering)

    def mod_lighting(img, steering, offset_saturation = -1, offset_lightness = -1):
        if offset_saturation == -1:
            offset_saturation = np.random.randn()*0.1;
        if offset_lightness == -1:
            offset_lightness  = np.random.randn()*0.3;
        img_hsv = matplotlib.colors.rgb_to_hsv(img/255.0);
        img_hsv[:,:,1] = np.maximum( np.minimum(img_hsv[:,:,1]+offset_saturation, 1.0), 0.0);
        img_hsv[:,:,2] = np.maximum( np.minimum(img_hsv[:,:,2]+offset_lightness, 1.0), 0.0);
        img_rgb = int(matplotlib.colors.hsv_to_rgb(img_hsv));
        return (img_rgb, steering)

    def mod_blur(img, steering):
        # requires python package "Pillow"
        blur = np.random.randn()*2;
        img_blur = scipy.misc.imfilter(img, 'smooth')
        return (img_blur, steering)

    def mod_shadow(img, steering):
        # TODO
        return (img_blur, steering)

    # ----------------------------------------------------------------------------------------
    def __init__(self):
        self.csv_file        = None;
        self.dataset         = None;
        self.basepath        = None;
        self.img_counter     = 0;
        self.img_list        = [];
        self.cam_list        = [];
        self.filter_list     = [];
        self.labels          = [];
        self.filter_switcher = {
            0: DataGenerator.mod_identity,
            1: DataGenerator.mod_lighting,
            2: DataGenerator.mod_blur,
            3: DataGenerator.mod_flip,
        }
        return

    def preprocess(self, basepath = "/mnt/data/", dataset = "dataset1_udacity"):
        self.csv_file = basepath + "/" + dataset + "/" + "driving_log.csv";
        self.dataset  = dataset;
        self.basepath = basepath;
        csv = pd.read_csv(self.csv_file)
        # center,left,right,steering,throttle,brake,speed

        # Strip whitespaces
        for i in csv.columns:
            if isinstance(csv[i][1], str):
                csv[i] = csv[i].map(str.strip)

        num_of_cams = 1;
        num_of_filters = len(self.filter_switcher);

        if isinstance(csv['left'][0], str): num_of_cams += 1;
        if isinstance(csv['right'][0], str): num_of_cams += 1;

        num_of_samples = len(csv) * num_of_cams * num_of_filters;

        if not os.path.exists(basepath + "/" + dataset + "/IMG_preprocessed/"):
            os.mkdir(basepath + "/" + dataset + "/IMG_preprocessed/");

        labels   = np.zeros(shape=[num_of_samples,4], dtype=np.float32)
        img_list = []
        cam_list = []
        filter_list = []

        img_counter = 0;
        for i in range(10): # range(len(csv)):
            left_img        = csv['left'][i];
            right_img       = csv['right'][i];
            center_img      = csv['center'][i];
            steering_angle  = csv['steering'][i];
            throttle        = csv['throttle'][i];
            brake           = csv['brake'][i];
            speed           = csv['speed'][i];

            for camera, img_path in (('L', left_img), ('R', right_img), ('C', center_img)):
                if not camera == 'C' and num_of_samples > 1:
                    continue

        #       EVTL: np.clip, np.min
                if camera == 'L': steering_angle = steering_angle + steering_offset;
                if camera == 'R': steering_angle = steering_angle - steering_offset;

                #print("loading " + img_path);
                img = cv2.imread(basepath + "/" + dataset + "/IMG/" + img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV does not use RGB, it uses BGR
                img = cv2.resize(img, (self.IMG_W, self.IMG_H))

                for idx, func_filter in self.filter_switcher.items():
                    (img_new, steering_new) = func_filter( img, steering_angle );

                    #img_path_full = dataset + "/IMG_preprocessed/" + camera + "_" + func_filter.__name__ + "_" +  img_path
                    img_path_full = dataset + "/IMG_preprocessed/" + func_filter.__name__ + "_" +  img_path
                    #print("saving:" + img_path_full);
                    #if False:
                    img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR);
                    cv2.imwrite(basepath + "/" + img_path_full , img_new)

                    img_list.append(img_path_full);
                    cam_list.append(camera);
                    filter_list.append(func_filter.__name__);
                    labels[img_counter,:] = (steering_new, throttle, brake, speed);
                    img_counter += 1;

        self.img_list    = img_list;
        self.cam_list    = cam_list;
        self.filter_list = filter_list;
        self.labels      = labels;
        self.img_counter = img_counter;

    # ----------------------------------------------------------------------------------------
    def save_index(self):
        dataset = list(zip(self.img_list,self.labels[:,0],self.labels[:,1], \
                  self.labels[:,2],self.labels[:,3], self.cam_list, self.filter_list ))
        df = pd.DataFrame(data = dataset, columns=['img', 'steering', 'throttle', 'brake', 'speed', 'cam', 'filter'])

        index_path = self.basepath + "/" + self.dataset + "/" + "index";

        df.to_csv(index_path + '.csv',mode='w')
        df.to_pickle(index_path + '.pkl')
        return df

    # ----------------------------------------------------------------------------------------
    def filter_data(remove_car_not_moving = True):
        if remove_car_not_moving:
            N = self.data.shape[0]
            self.data = self.data[self.data.Speed >= 10.]
            print("%d samples removed due to speed < 10" % (N - self.data.shape[0]))

    # ----------------------------------------------------------------------------------------
    def add_file(self, csv_file):
        # TODO: Allow csv_file list
        return

    # ----------------------------------------------------------------------------------------
    def calc_weights():
        """
        calculate higher probabilty for turns
        """
        return

    # ----------------------------------------------------------------------------------------
    def shuffle():
        return

    # ----------------------------------------------------------------------------------------
    def smooth_steering():
        return

    # ----------------------------------------------------------------------------------------
    def mirror():
        return

    # ----------------------------------------------------------------------------------------
    def split():
        #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0)
        features = None;
                #labels = None;

    # ----------------------------------------------------------------------------------------
    def __next__(self):
        self.index %= self.size()
        self.index += 1
        return self.log.iloc[self.index-1]

    def __iter__(self):
        return self

# datagen = DataGenerator("/mnt/data/dataset1_udacity")


#       features[img_counter,:,:,:] = img;
#       labels[img_counter] = steering_angle;
#       img_counter += 1;
#                    features[img_counter,:,:,:] = img_new;
            #left_img   = basepath + os.path.basename(csv['left'][i]);
            #right_img  = basepath + os.path.basename(csv['right'][i]);
            #center_img = basepath + os.path.basename(csv['center'][i]);
#        features = np.zeros(shape=[num_of_samples,IMG_H,IMG_W,3]);
