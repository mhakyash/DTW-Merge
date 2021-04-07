#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:08:30 2020

@author: mohammad
"""

import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import functions

original_path = os.path.realpath('') + '/UCRArchive_2018/'
list_of_folder = os.listdir(original_path)
list_of_folder = sorted(list_of_folder)

#be aware, whenever you run the code, the previous results will be deleted.
try: 
	os.remove('results_ResNet') 
	print("results_ResNet removed successfully") 
except: 
	print("results_ResNet can not be removed")

#=================================================================================================

for counter, folder_name in enumerate(list_of_folder):

	number_of_mul = 5

	#=================================================================================================

	#Reading the train samples
	train_samples, train_labels, num_classes = functions.utils.read_train_data(functions.utils, original_path, folder_name)

	#=================================================================================================

	#augmentation
	sample_label_list = []
	for (h, k) in zip(train_samples, train_labels):
		sample_label_list.append([h, k])

	augmented_dataset = functions.utils.aug_dataset(functions.utils, train_samples, train_labels, sample_label_list, number_of_mul = number_of_mul) 
	    
	augmented_dataset = np.array(augmented_dataset)  
	train_labels = np.array(train_labels)   

	#=================================================================================================

	#splitting train data
	skf = StratifiedKFold(n_splits=4, shuffle=True)

	for train_index, val_index in skf.split(augmented_dataset, train_labels):
		X_train, X_val = augmented_dataset[train_index], augmented_dataset[val_index]
		y_train, y_val = train_labels[train_index], train_labels[val_index]	

	augmented_dataset = np.array(X_train)  
	train_labels = np.array(y_train)   

	X_val = np.array(X_val)
	y_val = np.array(y_val)
    
	#=================================================================================================
	
	#Reading test samples
	test_samples, test_labels = functions.utils.read_test_data(functions.utils, original_path, folder_name)

	#=================================================================================================
	
	#create and training network    
	_, shape2 = np.shape(augmented_dataset)

	augmented_dataset = np.reshape(augmented_dataset, (-1, shape2 , 1))
	test_samples = np.reshape(test_samples, (-1, shape2 , 1))
	input_shape = (shape2, 1)
	X_val = np.reshape(X_val, (-1, shape2 , 1))

	model = functions.utils.build_model(input_shape = input_shape, num_classes = num_classes)

	model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                      metrics=['acc'])

	one_hot_encode = to_categorical(train_labels)
	one_hot_encode_test = to_categorical(test_labels)
	one_hot_encode_val = to_categorical(y_val)    

	reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                      patience=50, min_lr=0.0001) 

	checkpoint1 = ModelCheckpoint('val_loss.hdf5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
	checkpoint2 = ModelCheckpoint('train_loss.hdf5', save_best_only=True, save_weights_only=True, monitor='loss', mode='min')
      
	history = model.fit(augmented_dataset, one_hot_encode, epochs=10, batch_size=64, validation_data = (X_val ,one_hot_encode_val), callbacks = [reduce_lr, checkpoint1, checkpoint2])

	#=================================================================================================

	#saving the results
	model.load_weights('val_loss.hdf5')

	_, test_acc_min_val_loss = model.evaluate(test_samples, one_hot_encode_test)

	model.load_weights('train_loss.hdf5')

	_, test_acc_min_train_loss = model.evaluate(test_samples, one_hot_encode_test)

	with open("results_ResNet", "a+") as f:
		f.write("%d, %s, %f, %f\n" % (counter, folder_name, test_acc_min_val_loss, test_acc_min_train_loss))


