#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:58:33 2020

@author: mohammad
"""


import numpy as np
import os
from dtaidistance import dtw
import functions

original_path = 'UCRArchive_2018/'
list_of_folder = os.listdir(original_path)
list_of_folder = sorted(list_of_folder)


try: 
	os.remove('results_NN_DTW') 
	print("counter file removed successfully") 
except: 
	print("counter file can not be removed")
    
    
for counter, folder_name in enumerate(list_of_folder):
	

	number_of_mul = 2

	#=================================================================================================

	train_samples, train_labels, num_classes = functions.utils.read_train_data(functions.utils, original_path, folder_name)
		
	sample_label_list = []
		
	sample_label_list = []
	for (h, k) in zip(train_samples, train_labels):
		sample_label_list.append([h, k])

	augmented_dataset = functions.utils.aug_dataset(functions.utils, train_samples, train_labels, sample_label_list, number_of_mul = number_of_mul) 
	    
	augmented_dataset = np.array(augmented_dataset)  
	train_labels = np.array(train_labels)  
    
    
	test_samples, test_labels = functions.utils.read_test_data(functions.utils, original_path, folder_name)


	test_samples = np.array(test_samples)  


	test_samples = test_samples.astype('double')

	#calculating NN-DTW
	predicted_label = []
	for index, test_sample in enumerate(test_samples):
		print(index)
		distances = []
		for train_sample in train_samples:
			dtw_dis = dtw.distance_fast(test_sample, train_sample)
			distances.append(dtw_dis)
		predicted_label.append(train_labels[np.argmin(distances)])


	results = []
	for (x,y) in zip(predicted_label, test_labels):
		if x == y:
			results.append(1)
		else:
			results.append(0)

	final_result = 1 - (np.sum(results) / len(results))

	print(final_result)
    
	f = open("results_NN_DTW", "a+")
	    
	f.write("%d, %s, %f\n" % (counter, folder_name, final_result))

	f.close



