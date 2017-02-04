
python -m sdc.train_regression --train model.?? --nb_epoch 6 --dataset data1
python -m sdc.train_regression --train model.?? --nb_epoch 6 --dataset data1


"""
This is the main script to train the SDCRegression model implemented in `model.py`
Usage:
1. To build a model from scratch, run
```cmd
python -m sdc.train_regression --train --nb_epoch 6
```
where `sdc.train_regression` build a vgg16-based regression model from CenterImage to Steer. `--nb_epoch` is the number of training epoches.
2. To continuously train a model from a previous one, run
```cmd
python -m sdc.train_regression --train --restore --nb_epoch 6
```
3. To load a trained model to evaluate on test data without any training, simply run
```cmd
python -m sdc.train_regression --restore
```
"""
