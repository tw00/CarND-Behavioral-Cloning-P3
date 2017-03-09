
# README (Behavioral Cloning Project)

<span style='color:blue'>**NOTE TO REVIEWER:**</span>

<span style='color:blue'>As suggested I implemented the nvidia architecture as well (see model.py, line 320-341). After training the nvidia model, the car is now driving around the first track without leaving it. I also included a video.mp4.
</span>

# Project structure

## 1. Project files

My project includes the following files:
* self_driving_car/data_generator.py 
* self_driving_car/model.py  
* drive.py for driving the car in autonomous mode
* model.hdf5 containing a trained convolution neural network 

The `data_generator.py` contains two classes `DataPreprocessor` and `DataGenerator`. `DataPreprocessor` preprocesses the data and saves precomputed images in a separate `IMG_preprocessed` folder. Also, it creates an easy to process list of all samples (`index.pkl`). `DataGenerator` provides data during training, it also takes care of data balancing and filtering.

The model.py file contains a class `SDRegressionModel` for training and saving different convolution neural networks as well as input normalization. 

## 2. Running the model
Using the Udacity provided simulator (beta) and my drive.py file, the car can be driven autonomously by executing 

```sh
python drive.py model.hdf5
```

# Model Architecture and Training Strategy

## 1. Finding an appropriate model architecture

I archived the best results with `model_nvidia`. This model consists of three convolutional layers with 4x4 and 8x8 filter sizes and depths 16 and three fully connected layers (see `self_driving_car/model.py`, `model_nvidia`). The architecture is shown in the following image:

![alt text](./doc/images/model_nvidia.png)

The model includes LeakyReLU layers to introduce nonlinearity (model.py lines 251-263).
The data is normalized in the model using `SDRegressionModel.normalize` (model.py lines 66).

## 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 259-262). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). Training data is provided using a generator through the method
`DataGenerator.get_batch_generator` (`self_driving_car/data_generator.py`),
validation data is provided by the method
`DataGenerator.get_valid_data` (`self_driving_car/data_generator.py`).

## 3. Model parameter tuning

The model used an Nesterov Adam (nadam) optimizer, so the learning rate was not tuned automatically (model.py line 25). I tried different optimizers adam and nadam both gave good results.

## 4. Appropriate training data

For training data, I used data provided by Udacity as well as a few own laps. I also drove the track backward.
I also created numerous recordings of critical corners. This results in the following datasets:

```
dataset1_udacity      dataset4_beta_sim        dataset7_curve2B  dataset8_curve3C
dataset2_twe_one_lap  dataset5_beta_backwards  dataset8_curve3A  dataset8_curve3D
dataset3_ssz_one_lap  dataset6_curve2A         dataset8_curve3B
```

All training samples are kept in a table, that looks like (index.pkl):

![alt text](./doc/table.png)


# Model Architecture and Training Strategy

## 1. Solution Design Approach

My overall strategy includes the following steps:

1. Generating more data
2. Augmenting the data
3. Filtering and balancing the training data
4. Train a few epochs
5. Calculate accuracy using validation data
6. Testing the model on track 1
7. Go back to step 1 further improving the best model
(all steps are documented in `CarND_Behavioral_Cloning_Training_Part5.ipynb`)

I repeated this process for different architectures:

* The CommaAI architecture
* A modified CommaAI architecture with cropping and normalization
* A simple architecture with a low neuron count
* A modified 2nd simple architecture with more convolutional filters
* A modified 3rd simple architecture with more fully connected neurons
* A modified 4th simple architecture more tunable parameters
* The NVIDIA architecture

The reason why I created different models, was that I believed that I was not able to get a further improvement because the model complexity was too low so the model is not able to capture the complexity of the given task.

I implemented different methods that allow me to filter and balance training data. As it turns out that both steps are crucial for successful training. I, therefore, implemented the following methods for the `DataGenerator` class. 

* `add_dataset(self, dataset, basepath = '/mnt/data/')`
Allows to add (preprocessed) datasets, different datasets can be combined

* `filter_data_not_moving(self, not_moving_threshold = 10.)`
Removes frames at which the car is not moving or moving very slow

* `filter_data_low_steering(self, low_steering_threshold = 0.025, low_steering_remove_prop = 0.5)`
Removes samples with low steering angle with a given propability

* `smooth_steering(self, cam = 'C', window = 4)`
Smooths the steering angle

* `shuffle(self)`
Shuffle training data

* `correct_camera_steering(self, offset = 0.0)`
Correct the camera steering angle for left/right camera images

* `activate_mod(self, mod, dset='train')`
Activates precomputed modifications (image augmentations)

* `deactivate_cam(self, cam)`
Deactivates samples that were taken with a certain camera

* `split(self, valid_size=0.1)`
Splits data into training and validation dataset

Executing these methods in the correct order is important.

For data augmentation I implemented different filters:

* `DataPreprocessor.mod_identity` --> no modification
* `DataPreprocessor.mod_lighting` --> randomly modifies the lighting
* `DataPreprocessor.mod_blur`     --> blurs images
* `DataPreprocessor.mod_flip`     --> flips images
* `DataPreprocessor.mod_shadow`   --> randomly adds shadows

I always tried to keep the dataset balance, which means that it contains equal amounts of left and right steering samples.

The final step was to run the simulator to see how well the car was driving around track one.


## 2. Final Model Architecture

The model architecture that gives the best results (model.py lines 245 - 268) looks like:

```python
model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5, input_shape=(66,200,3)) )
model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(1))                                                        
```

The model consisted of a convolution neural network with the following layers and layer sizes:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_36 (Lambda)               (None, 66, 200, 3)    0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
convolution2d_96 (Convolution2D) (None, 33, 100, 24)   1824        lambda_36[0][0]                  
____________________________________________________________________________________________________
spatialdropout2d_96 (SpatialDrop (None, 33, 100, 24)   0           convolution2d_96[0][0]           
____________________________________________________________________________________________________
convolution2d_97 (Convolution2D) (None, 17, 50, 36)    21636       spatialdropout2d_96[0][0]        
____________________________________________________________________________________________________
spatialdropout2d_97 (SpatialDrop (None, 17, 50, 36)    0           convolution2d_97[0][0]           
____________________________________________________________________________________________________
convolution2d_98 (Convolution2D) (None, 7, 23, 48)     43248       spatialdropout2d_97[0][0]        
____________________________________________________________________________________________________
spatialdropout2d_98 (SpatialDrop (None, 7, 23, 48)     0           convolution2d_98[0][0]           
____________________________________________________________________________________________________
convolution2d_99 (Convolution2D) (None, 5, 21, 64)     27712       spatialdropout2d_98[0][0]        
____________________________________________________________________________________________________
spatialdropout2d_99 (SpatialDrop (None, 5, 21, 64)     0           convolution2d_99[0][0]           
____________________________________________________________________________________________________
convolution2d_100 (Convolution2D (None, 3, 19, 64)     36928       spatialdropout2d_99[0][0]        
____________________________________________________________________________________________________
spatialdropout2d_100 (SpatialDro (None, 3, 19, 64)     0           convolution2d_100[0][0]          
____________________________________________________________________________________________________
flatten_20 (Flatten)             (None, 3648)          0           spatialdropout2d_100[0][0]       
____________________________________________________________________________________________________
dropout_39 (Dropout)             (None, 3648)          0           flatten_20[0][0]                 
____________________________________________________________________________________________________
dense_77 (Dense)                 (None, 100)           364900      dropout_39[0][0]                 
____________________________________________________________________________________________________
dense_78 (Dense)                 (None, 50)            5050        dense_77[0][0]                   
____________________________________________________________________________________________________
dense_79 (Dense)                 (None, 10)            510         dense_78[0][0]                   
____________________________________________________________________________________________________
dropout_40 (Dropout)             (None, 10)            0           dense_79[0][0]                   
____________________________________________________________________________________________________
dense_80 (Dense)                 (None, 1)             11          dropout_40[0][0]                 
====================================================================================================
Total params: 501,819
Trainable params: 501,819
Non-trainable params: 0
____________________________________________________________________________________________________
```

I tried different architecture and 160k trainable parameters should be enough since I already came up with good results with only 18k parameters.


## 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded different laps on track one using center lane driving.
Fist the images are resized and augmented, which looks like (here blurring and shadow):

![alt text](./doc/blur.png)
![alt text](./doc/shadow.png)

For most model architecture the image is cropped, which gives an image of the relevant area:

![alt text](./doc/cropping.png)

After applying image augmentation and filtering samples, the training data looks well balanced.

![alt text](./doc/table.png)

This is the data that I used for the first 14 training epochs:

![alt text](./doc/data1.png)

The initial model I then train with more data:

![alt text](./doc/data2.png)

For the initial training both training error and validation error decrease:

![alt text](./doc/loss1.png)

Later no big change in the validation error is seen.

# Discussion

During this project I tried different things to get a better model:

* getting more training data
* getting more training data of critical curves
* increase model complexity
* decrease model complexity
* implement methods to prevent overfitting
* train for more epochs (I trained for +4h an AWS)
* augment data
* removing low steering
* cropping irrelevant data
* different steering angle correction for left/right camera
* initialization, different optimizers, different activation functions

In order to get further improvement more augmentation and data from different tracks could be used. This could produce a more general model.
