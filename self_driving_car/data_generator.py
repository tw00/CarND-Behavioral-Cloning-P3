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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from self_driving_car import yuv_colorspace # TWE

class DataPreprocessor:
    """
    This class serves the following purposes:
        Creating a collection of preprocessed images (IMG_preprocessed)
        and a training data base (index.pkl)
    """
    # orginal size: 320x160 RGB 8-bit
    IMG_W = 192;
    IMG_H = 192;
    steering_offset = 0; # done later...

    # ----------------------------------------------------------------------------------------
    def mod_identity(img, steering):
        #return (image, steering angle, fliped?)
        return (img, steering, 1)

    def mod_flip(img, steering):
        img  = cv2.flip(img,1);
        steering = -steering;
        return (img, steering, -1)

    def mod_lighting(img, steering, offset_saturation = -1, offset_lightness = -1):
        if np.random.rand() > 0.5:
            # keep data balanced
            (img, steering, flip) = DataPreprocessor.mod_flip(img, steering)
        else: flip = 1
        if offset_saturation == -1:
            offset_saturation = np.random.randn()*0.1;
        if offset_lightness == -1:
            offset_lightness  = np.random.randn()*0.3;
        img_hsv = matplotlib.colors.rgb_to_hsv(img.astype(np.float32)/255.0);
        img_hsv[:,:,1] = np.maximum( np.minimum(img_hsv[:,:,1]+offset_saturation, 1.0), 0.0);
        img_hsv[:,:,2] = np.maximum( np.minimum(img_hsv[:,:,2]+offset_lightness, 1.0), 0.0);
        img_rgb = matplotlib.colors.hsv_to_rgb(img_hsv)*255;
        return (img_rgb.astype(np.uint8), steering, flip)

    def mod_shadow(img, steering, offset_shadow = 0.35, minx = 30, miny=30):
        if np.random.rand() > 0.5:
            # keep data balanced
            (img, steering, flip) = DataPreprocessor.mod_flip(img, steering)
        else: flip = 1
        img_hsv = matplotlib.colors.rgb_to_hsv(img.astype(np.float32)/255.0);
        x = np.random.randint(0,img.shape[0]-minx)
        y = np.random.randint(0,img.shape[1]-miny)
        dx = np.random.randint(minx,img.shape[0]-x)
        dy = np.random.randint(miny,img.shape[1]-y)
        shadow_coords = ((x,x+dx),(y,y+dy))
        #print("shadow_coords", shadow_coords)
        img_hsv[shadow_coords[0][0]:shadow_coords[0][1],shadow_coords[1][0]:shadow_coords[1][1],2] = \
            np.maximum( \
                np.minimum( \
                    img_hsv[shadow_coords[0][0]:shadow_coords[0][1],shadow_coords[1][0]:shadow_coords[1][1],2] \
                    - offset_shadow, 1.0), 0.0);
        img_rgb = matplotlib.colors.hsv_to_rgb(img_hsv)*255;
        return (img_rgb.astype(np.uint8), steering, flip)

    def mod_blur(img, steering):
        if np.random.rand() > 0.5:
            # keep data balanced
            (img, steering, flip) = DataPreprocessor.mod_flip(img, steering)
        else: flip = 1
        img_blur = scipy.misc.imfilter(img, 'smooth')
        return (img_blur, steering, flip)

    # ----------------------------------------------------------------------------------------
    def __init__(self):
        self.csv_file        = None;
        self.dataset         = None;
        self.basepath        = None;
        self.img_counter     = 0;
        self.img_list        = [];
        self.flip_list       = [];
        self.cam_list        = [];
        self.filter_list     = [];
        self.labels          = [];
        self.filter_switcher = {
            0: DataPreprocessor.mod_identity,
            1: DataPreprocessor.mod_lighting,
            2: DataPreprocessor.mod_blur,
            3: DataPreprocessor.mod_flip,
            4: DataPreprocessor.mod_shadow,
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
        flip_list = []
        filter_list = []

        img_counter = 0;
#        for i in range(10):
        for i in range(len(csv)):
            left_img        = csv['left'][i];
            right_img       = csv['right'][i];
            center_img      = csv['center'][i];
            steering_angle  = csv['steering'][i];
            throttle        = csv['throttle'][i];
            brake           = csv['brake'][i];
            speed           = csv['speed'][i];

            for camera, img_path in (('L', left_img), ('R', right_img), ('C', center_img)):
                if not camera == 'C' and num_of_cams == 1:
                    continue

        #       EVTL: np.clip, np.min
                if camera == 'L': steering_angle = steering_angle + DataPreprocessor.steering_offset;
                if camera == 'R': steering_angle = steering_angle - DataPreprocessor.steering_offset;

                #print("loading " + img_path);
                img = plt.imread(basepath + "/" + dataset + "/IMG/" + img_path);
                img = cv2.resize(img, (self.IMG_W, self.IMG_H))

                for idx, func_filter in self.filter_switcher.items():
                    (img_new, steering_new, flip) = func_filter( img, steering_angle );

                    img_path_full = dataset + "/IMG_preprocessed/" + func_filter.__name__ + "_" +  img_path
                    #print("saving:" + img_path_full);
                    plt.imsave(basepath + "/" + img_path_full , img_new)

                    img_list.append(img_path_full);
                    cam_list.append(camera);
                    flip_list.append(flip);
                    filter_list.append(func_filter.__name__);
                    labels[img_counter,:] = (steering_new, throttle, brake, speed);
                    img_counter += 1;

        self.img_list    = img_list;
        self.cam_list    = cam_list;
        self.flip_list   = flip_list;
        self.filter_list = filter_list;
        self.labels      = labels;
        self.img_counter = img_counter;

    # ----------------------------------------------------------------------------------------
    def save_index(self):
        dataset = list(zip(self.img_list,self.labels[:,0],self.labels[:,1], \
                  self.labels[:,2],self.labels[:,3], self.cam_list, self.filter_list, self.flip_list ))
        df = pd.DataFrame(data = dataset, columns=['img', 'steering', 'throttle', 'brake', 'speed', 'cam', 'filter', 'flip'])

        index_path = self.basepath + "/" + self.dataset + "/" + "index";

        df.to_csv(index_path + '.csv',mode='w')
        df.to_pickle(index_path + '.pkl')
        return df

# ============================================================================================
class DataGenerator:
    """
    This class serves the following purposes:
        Generate training and validation data on the fly
    """


    # ----------------------------------------------------------------------------------------
    def __init__(self):
        self.data        = None;
        self.normalizer  = None;
        self.datasets    = [];
        self.size        = 0;
        self.index       = 1;
        self.basepath    = "./"
        self.img_loaded_to_ram = False;
        self.img_cache   = {}

    # ----------------------------------------------------------------------------------------
    def add_dataset(self, dataset, basepath = '/mnt/data/', use_original_data = False):
        self.datasets.append(dataset);
        self.basepath = basepath # HACK
        index_path = basepath + "/" + dataset + "/" + "index";
        df = pd.read_pickle(index_path + '.pkl')
        if use_original_data:
            replacer = lambda x: x.replace('IMG_preprocessed', 'IMG').replace('mod_identity_',
                                                                              '').replace('mod_flip_',
                                                                                          '')
            df['img'] = df['img'].apply(replacer)
        if isinstance(self.data,pd.DataFrame):
            self.data = pd.concat([self.data, df], ignore_index=True)
        else:
            self.data = df;

    # ----------------------------------------------------------------------------------------
    def load_img_to_ram(self, flip = False):
        print('loading images to ram...');
        valid_rows = self.data[self.data['is_active']]
        self.img_loaded_to_ram = False;
        self.img_cache = dict();
        for i, row in valid_rows.iterrows():
            name = row['img']
            img = self.read_img(name);
            if flip and row['filter'] == 'mod_flip':
                imgA = cv2.flip(img,1);
            self.img_cache[name] = img;
        self.img_loaded_to_ram = True;
        print('all images loaded to ram');

    # ----------------------------------------------------------------------------------------
    def auto_prepare(self, filter_data = True, smooth_steering = True, shuffle = True):
        if smooth_steering:
            self.smooth_steering();
        if shuffle:
            self.shuffle();
        if filter_data:
            self.filter_data_not_moving();
            self.filter_data_low_steering();
        self.calc_weights();
        self.split();

    # ----------------------------------------------------------------------------------------
    def filter_data_not_moving(self, not_moving_threshold = 10.):
        N = self.num_of_samples('all');
        self.data = self.data[self.data.speed >= not_moving_threshold]
        print("%d samples removed due to speed < %f" % (N - self.data.shape[0], not_moving_threshold))

    # ----------------------------------------------------------------------------------------
    def filter_data_low_steering(self, low_steering_threshold = 0.025, low_steering_remove_prop = 0.5):
        N = self.num_of_samples('all');
        idx_low_steering = (np.abs(self.data.steering) < low_steering_threshold).values
        idx_low_sparse = np.where(idx_low_steering == True, np.random.rand(len(idx_low_steering)) < low_steering_remove_prop, False)
        self.data = self.data[np.logical_not(idx_low_sparse)];
        print("%d samples randomly removed due to steering < %f" % (N - self.data.shape[0], low_steering_threshold))

    # ----------------------------------------------------------------------------------------
    def smooth_steering(self, cam = 'C', window = 4):
        # FutureWarning: pd.ewm_mean is deprecated for Series and will be removed in a future
        # version, replace with
        #    Series.ewm(min_periods=0,span=10,adjust=True,ignore_na=False).mean()
        steering = self.data.loc[(self.data["cam"] == cam) & (self.data["filter"] == "mod_identity"), "steering"].values

        fwd = pd.stats.moments.ewma( steering, span=window )
        bwd = pd.stats.moments.ewma( steering[::-1], span=window )
        steering_smooth = np.mean( np.vstack(( fwd, bwd[::-1] )), axis=0 )
        print(np.mean(steering));
        print(np.mean(steering_smooth));

        for mod in ("mod_identity", "mod_flip", "mod_shadow", "mod_lighting", "mod_blur"):
            flip = self.data.loc[(self.data["cam"] == cam) & (self.data["filter"] == mod), "flip"]
            self.data.loc[(self.data["cam"] == cam) & (self.data["filter"] == mod), "steering"] = steering_smooth * flip
        # TODO: Smooth other camera perspectives?

        print("steering angle has been smoothed based on window %d" % window)

    # ----------------------------------------------------------------------------------------
    def calc_weights(self, p_min = 0.3, p_max = 1.0):
        """
        calculate higher probabilty for turns
        """
        sLength = len(self.data)
        # self.data['weight'] = np.ones((sLength,1),dtype=float)/sLength
        self.data['weight'] = self.data['steering'].map(lambda x: p_min + x*(p_max-p_min)/0.5)
        self.data['weight'] /= self.data['weight'].sum()
        print("higher probabilities for steering samples (sum of p = %f)" % self.data['weight'].sum())
        # TODO: Add data based on weights at the end (alternative)

    # ----------------------------------------------------------------------------------------
    def shuffle(self):
        self.data = self.data.reindex(np.random.permutation(self.data.index))
        print("shuffled data")

    # ----------------------------------------------------------------------------------------
    def split(self, valid_size=0.1):
        sLength = len(self.data)
        assert valid_size > 0 and valid_size < 1, "Invalid validation size"
        sLengthValid = round(sLength * valid_size)
        is_train = np.ones([sLength],dtype=bool)
        is_train[0:sLengthValid] = False
        self.data['is_train'] = is_train;
        self.data['is_valid'] = np.logical_not(is_train);
        self.data['is_active'] = False;
        # activate mod_identity by default
        self.activate_mod('mod_identity', dset='valid')
        self.activate_mod('mod_identity', dset='train')
        print("split data into %d training sample and %d validation samples" %
              (self.num_of_samples('train'), self.num_of_samples('valid')))
        return

    # ----------------------------------------------------------------------------------------
    def correct_camera_steering(self, offset = 0.0):
        self.data.loc[self.data['cam'] == 'R', 'steering'] -= offset;
        self.data.loc[self.data['cam'] == 'L', 'steering'] += offset;
        print("steering angle corrected by +/- %f" % offset)

    # ----------------------------------------------------------------------------------------
#    def deactivate_mod(self, mod):
#        # mod_identity, mod_lighting, mod_blur, mod_flip, mod_shadow,
#        self.data.loc[self.data['filter'] == mod, 'is_active'] = False;

    # ----------------------------------------------------------------------------------------
    def activate_mod(self, mod, dset='train'):
        # INFO: Activates all cams
        # mod_identity, mod_lighting, mod_blur, mod_flip, mod_shadow,
        self.data.loc[np.logical_and(self.data['filter'] == mod, self.data['is_'+dset]), 'is_active'] = True;

    # ----------------------------------------------------------------------------------------
    def deactivate_cam(self, cam):
        # L, R, C
        self.data.loc[self.data['cam'] == cam, 'is_active'] = False;

    # ----------------------------------------------------------------------------------------
#    def activate_cam(self, mod, dset = 'train'):
#        # L, R, C
#        self.data.loc[np.logical_and(self.data['cam'] == cam, self.data['is_'+dset]), 'is_active'] = True;

    # ----------------------------------------------------------------------------------------
    def num_of_samples(self, sample_set='train'): # sample_set = {all, train, valid}
        if sample_set == 'train':
            return np.sum(np.logical_and(self.data['is_active'], self.data['is_train'] ) == True)
        elif sample_set == 'valid':
            return np.sum(np.logical_and(self.data['is_active'], self.data['is_valid'] ) == True)
        elif sample_set == 'active':
            return np.sum(self.data['is_active'] == True)
        elif sample_set == 'all':
            # N = self.data.shape[0]
            return len(self.data)

    # ----------------------------------------------------------------------------------------
    def plot_stats( self ):
        x1 = self.data['steering'].values
        x2 = self.data[np.logical_and(self.data['is_train'], self.data['is_active'])]['steering'].values
        x3 = self.data[np.logical_and(self.data['is_valid'], self.data['is_active'])]['steering'].values

        print("Number of samples:           ", self.num_of_samples('all'))
        print("Number of active sample:     ", self.num_of_samples('active'))
        print("Number of training samples:  ", self.num_of_samples('train'))
        print("Number of validation samples:", self.num_of_samples('valid'))

        # plot histogram of the data
        fig = plt.figure(figsize=(18,4))
        plt.subplot(1,3,1)
        plt.title(r'Histogram of steeringe angles (all) (m = %.4f)' % np.mean(x1)); plt.grid(zorder=0)
        n, bins, patches = plt.hist(x1, 29, range=[-0.35, 0.35], normed=1, edgecolor='black', facecolor='green', alpha=0.75)
        plt.xlabel('Steering'); plt.ylabel('Probability')
        plt.xlim((-0.35,0.35)); plt.ylim((0,10));
        plt.subplot(1,3,2)
        plt.title(r'Histogram of steeringe angles (train) (m = %.4f)' % np.mean(x2)); plt.grid(True);
        n, bins, patches = plt.hist(x2, 29, range=[-0.35, 0.35], normed=1, edgecolor='black', facecolor='red', alpha=0.75)
        plt.xlabel('Steering'); plt.ylabel('Probability')
        plt.xlim((-0.35,0.35)); plt.ylim((0,10));
        plt.subplot(1,3,3)
        plt.title(r'Histogram of steeringe angles (valid) (m = %.4f)' % np.mean(x3)); plt.grid(True)
        n, bins, patches = plt.hist(x3, 29, range=[-0.35, 0.35], normed=1, edgecolor='black', facecolor='blue', alpha=0.75)
        plt.xlabel('Steering'); plt.ylabel('Probability')
        plt.xlim((-0.35,0.35)); plt.ylim((0,10)); 
        plt.show()

    # ----------------------------------------------------------------------------------------
    def get_batch_generator(self, batch_size = 128):
        # TODO Doesn't match with num_of_samples('train')
        def _generator(stream):
            batch_items = []
            for i, row in enumerate(stream):
                item = (row[0], row[1])
                batch_items.append(item)
                if len(batch_items) >= batch_size:
                    current_batch = batch_items[:batch_size]
                    yield tuple(map(np.asarray, zip(*current_batch)))
                    batch_items = batch_items[batch_size:]
        return _generator(self)

    # ----------------------------------------------------------------------------------------
    def get_valid_data(self):
        # TODO: consider mod_identity only
        valid_rows = self.data[np.logical_and(self.data['is_active'], self.data['is_valid'])]
        batch_items = []
        for i, row in valid_rows.iterrows():
            img = self.read_img(row['img'])
            img = self.normalizer(img);
            item = (img, row['steering'])
            batch_items.append(item)
        return tuple(map(np.asarray, zip(*batch_items)))

    # ----------------------------------------------------------------------------------------
    def __next__(self):
        # TODO: select based on weight
        self.index %= self.num_of_samples('train')
        self.index += 1
        row = self.data[np.logical_and(self.data['is_active'], self.data['is_train'])].iloc[self.index-1]
        img = self.read_img(row['img'])
        img = self.normalizer(img);
        return (img, row['steering'])

    # ----------------------------------------------------------------------------------------
    def __iter__(self):
        return self

    # ----------------------------------------------------------------------------------------
    def reset(self):
        self.index = 1
        return self

    # ----------------------------------------------------------------------------------------
    def read_img(self, name):
        if not self.img_loaded_to_ram:
            return plt.imread(self.basepath + "/" + name);
        else:
            return self.img_cache[name];
