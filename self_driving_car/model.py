# -*- coding: utf-8 -*-

import os
import pickle
import sys
import cv2

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# Build a model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
#from keras import callbacks

#from self_driving_car import yuv_colorspace

class SDRegressionModel():

    # ----------------------------------------------------------------------------------------
    def model_architecture(model):
        if model == "commaAI":
            return {'name': "commaAI",
                    'model': SDRegressionModel.model_commaAI(),
                    'normalizer': SDRegressionModel.normalize_comma}
        elif model == "commaAI_modified":
            return {'name': "commaAI_modified",
                    'model': SDRegressionModel.model_commaAI_modified(),
                    'normalizer': SDRegressionModel.normalize}
        elif model == "simple":
            return {'name': "simple",
                    'model': SDRegressionModel.model_simple(),
                    'normalizer': SDRegressionModel.normalize}
        elif model == "simple2":
            return {'name': "simple2",
                    'model': SDRegressionModel.model_simple2(),
                    'normalizer': SDRegressionModel.normalize2}
        elif model == "udacity":
            return {'name': "udacity",
                    'model': SDRegressionModel.model_udacity(),
                    'normalizer': SDRegressionModel.no_normalize}
        else: print("model %s not found" % model);

    # ----------------------------------------------------------------------------------------
    def normalize(img):
        img = img.copy();
        if img.shape != (128,128):
            img = cv2.resize(img, (128, 128))
        img = img.astype(float) / 255.0
        img = yuv_colorspace.rgb2yuv(img)   # convert to YUV colorspace
        img[:,:,0] = img[:,:,0] - 0.5;      # remove mean
        return img

    def denormalize(img):
        img = img.copy()
        img[:,:,0] = img[:,:,0] + 0.5;
        img = self_driving_car.yuv_colorspace.yuv2rgb(img)
        img = img * 255
        return img.astype(np.uint8)

    def normalize2(img):
        img = img.copy();
        if img.shape != (192,192):
            img = cv2.resize(img, (192, 192))
        img = img.astype(float) / 255.0
        img = yuv_colorspace.rgb2yuv(img)   # convert to YUV colorspace
        img[:,:,0] = img[:,:,0] - 0.5;      # remove mean
        return img

    def normalize_comma(img):
        img = img.copy();
        if img.shape != (160,320):
            img = cv2.resize(img, (160, 320))
#        img = img.astype(float) / 255.0
#        img = yuv_colorspace.rgb2yuv(img)   # convert to YUV colorspace
#        img = img.astype * 255
        return img

    def no_normalize(img):
        if img.shape != (192,192):
            img = img.copy();
            img = cv2.resize(img, (192, 192))
        return img

    # ----------------------------------------------------------------------------------------
    def model_crop():
        model = Sequential()
        model.add(Cropping2D(cropping=((45,15),(0,0)), input_shape=(128,128,3)))
        model.compile(optimizer="adam", loss="mse")
        return model

    # ----------------------------------------------------------------------------------------
    def model_commaAI():
        ch, row, col = 3, 320, 160  # camera format
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    # ----------------------------------------------------------------------------------------
    def model_commaAI_modified():
        # Input 128x128x3 YUV normalized!
#        ch, row, col = 3, 128, 128  # camera format
        use_dropout = True

        def crop(img):
            #return img[45:-15,:,:]
#            return img[:, 45:-15, :]
            return img[45:-15, :, :]

        model = Sequential()
        #model.add(Lambda(crop, input_shape=(128, 128, 3), name="crop"))
        model.add(Cropping2D(cropping=((45,15),(0,0)), input_shape=(128,128,3)))
        #model.add(Lambda(lambda x: x, input_shape=(128, 128, 3)))
        #model.add(Lambda(normalize, name="normalize"))

        with tf.name_scope('conv2D_1'):
            #model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
            model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        with tf.name_scope('conv2D_2'):
            model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        with tf.name_scope('conv2D_3'):
            model.add(Convolution2D(64, 5, 5, subsample=(4, 4), border_mode="same"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        if use_dropout: # NEU
            model.add(Dropout(.2)) # NEU
        model.add(ELU())
        model.add(Dense(512))
        if use_dropout:
            model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(20))
        model.add(ELU())
        model.add(Dense(1))

        # TODO: LR = 0.0001
        model.compile(optimizer="adam", loss="mse")
        # optimizer = Adam(lr=learning_rate)
        # model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # model.summary()

        return model

    # ----------------------------------------------------------------------------------------
    def model_simple():
        # Input 128x128x3 YUV normalized!

        model = Sequential()
        model.add(Cropping2D(cropping=((45,15),(0,0)), input_shape=(128,128,3)))
        model.add(Convolution2D(32, 2, 2, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(Convolution2D(64, 2, 2, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(MaxPooling2D())
        model.add(Convolution2D(2, 2, 2, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(Convolution2D(4, 2, 2, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
#        model.add(MaxPooling2D()) # ENTFERNT
        model.add(Flatten())
        #model.add(Dense(256))
        #model.add(Dense(64))
#        model.add(Dropout(.2))  # NEU
#        model.add(LeakyReLU())  # NEU
        model.add(Dense(128))
        model.add(Dropout(.5))
        model.add(LeakyReLU())
        model.add(Dense(32))
        model.add(Dropout(.2))
#        model.add(LeakyReLU()) # NEU 2
#        model.add(Dense(10))   # NEU 2
        model.add(LeakyReLU())
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")
        return model

    # ----------------------------------------------------------------------------------------
    def model_simple2():
        model = Sequential()
        model.add(Cropping2D(cropping=((65,20),(0,0)), input_shape=(192,192,3)))
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(MaxPooling2D())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dropout(.5))
        model.add(LeakyReLU())
        model.add(Dense(64))
        model.add(Dropout(.2))
        model.add(Dense(10))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")
        return model

    def model_udacity():
        # Input RAW
        model = Sequential()
        model.add(Lambda( lambda x: x / 255.0 - 0.5, input_shape=(192,192,3) ))
        model.add(Cropping2D(cropping=((65,20),(0,0))))
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Convolution2D(16, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")
        return model

    def __init__(self, model = 'commaAI_modified', basepath = '/mnt/models/'):
        model_info      = SDRegressionModel.model_architecture(model)
        self.model      = model_info['model'];
        self.normalizer = model_info['normalizer'];
        self.modelname  = model_info['name'];
        self.basepath   = basepath;
        self._history   = None
        self._last_session_name = None

    # ----------------------------------------------------------------------------------------
    def train_simple(self, X_train_norm, y_train_norm):
        # Working with large datasets like Imagenet #68
        # https://github.com/fchollet/keras/issues/68

        #model = model_nvidia([64,128,3])
        model = self.model
        history = model.fit(X_train_norm, y_train_norm, nb_epoch=100, validation_split=0)
        return history

    # ----------------------------------------------------------------------------------------
    def train( train_existing_model = False):
        if train_existing_model:
            self.restore_model(path)

    # ----------------------------------------------------------------------------------------
    def train_generator( self, datagen, session_name, nb_epoch = 10, lr = 0, samples_per_epoch = 0):
        model = self.model;
        self._last_session_name = session_name;
        datagen.normalizer = self.normalizer

        if not lr == 0:
            print("setting learning rate to %f" % lr );
            model.optimizer.lr.assign(lr);
        if not os.path.exists(self.basepath + "/" + self.modelname):
            os.mkdir(self.basepath + "/" + self.modelname );
        if not os.path.exists(self.basepath + "/" + self.modelname + "/weights"):
            os.mkdir(self.basepath + "/" + self.modelname + "/weights" );
        if not os.path.exists(self.basepath + "/" + self.modelname + "/weights/" + session_name):
            os.mkdir(self.basepath + "/" + self.modelname + "/weights/" + session_name );
        if not os.path.exists(self.basepath + "/" + self.modelname + "/tb_log"):
            os.mkdir(self.basepath + "/" + self.modelname + "/tb_log" );
        if samples_per_epoch == 0:
            samples_per_epoch = datagen.num_of_samples('train');

        checkpoint_callback = ModelCheckpoint(self.basepath + "/" + self.modelname + "/weights/" +
                                              session_name + "/weights.{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1);
        tensorboard_callback = TensorBoard(log_dir=self.basepath + "/" + self.modelname + "/tb_log/");

#        gen_train = datagen.get_batch_generator('train');
        gen_train = datagen.get_batch_generator();
#        gen_valid = datagen.get_batch_generator('valid');
        valid_data = datagen.get_valid_data();

        # Fit the model on the batches generated datage generator
        with tf.name_scope('train'):
            history = model.fit_generator(gen_train,
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            validation_data=valid_data,
                            callbacks=[checkpoint_callback, tensorboard_callback]);
            # TODO: Save trainig history
        self._history = history;

    # ----------------------------------------------------------------------------------------
    def load_weights(self, session_name, epoch):
        path = self.basepath + "/" + self.modelname + "/weights/" + session_name + "/";
        result = [e for e in os.listdir(path) if e.startswith('weights.%02d-' % epoch)]
        model.load_weights( path + result[0] );

    # ----------------------------------------------------------------------------------------
    def save_history(self):
        with open(self.basepath + "/" + self.modelname + "/weights/" + self._last_session_name + "/history.pkl", "w") as f:
            pickle.dump(history, f)

    # ----------------------------------------------------------------------------------------
    def save_model_architecture(self, filename="model.json"):
        model = self.model
        with open(filename, "w") as f:
            json.dump(model.to_json(), f)

    # ----------------------------------------------------------------------------------------
    def save(self, prefix):
        """save model for future inspection and continuous training
        """
        model_file = prefix + ".json"
        weight_file = prefix + ".h5"
        json.dump(self.model.to_json(), open(model_file, "w"))
        self.model.save_weights(weight_file)
        return self

    # ----------------------------------------------------------------------------------------
    def restore_model(self, prefix):
        """restore a saved model
        """
        model_file = prefix + ".json"
        weight_file = prefix + ".h5"
        self.model = model_from_json(json.load(open(model_file)))
        self.model.load_weights(weight_file)
        return self
        model.load_weights("./output/model.h5")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide model name');
    else:
        model = SDRegressionModel.model_architecture(sys.argv[1])['model']
        model.summary()

#def checkpoint(filepattern="model.{epoch:02d}.h5"):
#return ModelCheckpoint(filepattern, save_weights_only=True)

#def preprocess_img(img, fromFile = False, fromDriver = True):
#    IMG_W = 128
#    IMG_H = 64
    #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #img = img.convert('RGB')
#    img = np.array(img)
    #print(img);
    #print(img.shape);
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV does not use RGB, it uses BGR
#    img = cv2.resize(img, (IMG_W, IMG_H))
#    img = img.astype(float)/255.0
#    img = yuv_colorspace.rgb2yuv(img) # convert to YUV colorspace
#    img[:,:,0] = img[:,:,0] - 0.5; # remove mean
#    return img
