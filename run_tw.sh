#/bin/bash

python feature_extraction.py --training_file inception_cifar10_100_bottleneck_features_train.p --validation_file inception_cifar10_bottleneck_features_validation.p --batch_size 100 --epochs 50
python feature_extraction.py --training_file resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet_cifar10_bottleneck_features_validation.p --batch_size 100 --epochs 50
python feature_extraction.py --training_file vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg_cifar10_bottleneck_features_validation.p --batch_size 100 --epochs 50

python feature_extraction.py --training_file inception_traffic_100_bottleneck_features_train.p --validation_file inception_traffic_bottleneck_features_validation.p --batch_size 100 --epochs 50
python feature_extraction.py --training_file resnet_traffic_100_bottleneck_features_train.p --validation_file resnet_traffic_bottleneck_features_validation.p --batch_size 100 --epochs 50
python feature_extraction.py --training_file vgg_traffic_100_bottleneck_features_train.p --validation_file vgg_traffic_bottleneck_features_validation.p --batch_size 100 --epochs 50

#Epoch 50/50
#1000/1000 [==============================] - 0s - loss: 0.0384 - acc: 1.0000 - val_loss: 1.1042 - val_acc: 0.6579

#Epoch 50/50
#1000/1000 [==============================] - 0s - loss: 0.0314 - acc: 1.0000 - val_loss: 0.8312 - val_acc: 0.7356

#Epoch 50/50
#1000/1000 [==============================] - 0s - loss: 0.0931 - acc: 0.9970 - val_loss: 0.8402 - val_acc: 0.7414

#Epoch 50/50
#4300/4300 [==============================] - 0s - loss: 0.0102 - acc: 1.0000 - val_loss: 0.8444 - val_acc: 0.7551

#Epoch 50/50
#4300/4300 [==============================] - 0s - loss: 0.0119 - acc: 1.0000 - val_loss: 0.6078 - val_acc: 0.8146

#Epoch 50/50
#4300/4300 [==============================] - 0s - loss: 0.0348 - acc: 0.9986 - val_loss: 0.3941 - val_acc: 0.8804

# inception_cifar10_100_bottleneck_features_train.p
# inception_cifar10_bottleneck_features_validation.p
# inception_traffic_100_bottleneck_features_train.p
# inception_traffic_bottleneck_features_validation.p
# resnet_cifar10_100_bottleneck_features_train.p
# resnet_cifar10_bottleneck_features_validation.p
# resnet_traffic_100_bottleneck_features_train.p
# resnet_traffic_bottleneck_features_validation.p
# vgg_cifar10_100_bottleneck_features_train.p
# vgg_cifar10_bottleneck_features_validation.p
# vgg_traffic_100_bottleneck_features_train.p
# vgg_traffic_bottleneck_features_validation.p
