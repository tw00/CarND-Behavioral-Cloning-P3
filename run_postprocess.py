#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf
import self_driving_car
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('dataset', '', "Name of dataset")

def main(_):
    preprocessor = self_driving_car.data_generator.DataPreprocessor()
    preprocessor.preprocess(basepath = "/mnt/data/", dataset = FLAGS.dataset)
    df = preprocessor.save_index()
    print("done");

# parses flags and calls the `main` function above
if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        print("Usage: run_postprocess.py --dataset <dataset>")
        sys.exit(1)
    tf.app.run()

#preprocessor.preprocess(basepath = "/mnt/data/", dataset = "dataset1_udacity")
#preprocessor.preprocess(basepath = "/mnt/data/", dataset = "dataset2_twe_one_lap")
#preprocessor.preprocess(basepath = "/mnt/data/", dataset = "dataset3_ssz_one_lap")
#load bottleneck data
#tf.app.run()
#flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
#flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
#flags.DEFINE_integer('batch_size', 256, "The batch size.")
#flags.DEFINE_integer('epochs', 50, "The number of epochs.")
