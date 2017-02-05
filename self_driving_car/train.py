#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import self_driving_car

datagen = self_driving_car.data_generator.DataGenerator()
datagen.add_dataset("dataset1_udacity")
datagen.prepare();


gen = datagen.get_batch_generator()
