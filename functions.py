import numpy as np
import tensorflow.keras
import fastdtw
import random
import pandas as pd
import tensorflow as tf


class utils:

	#random padding
	def padding(seq, size_of_series):
	    
	    mod_seq = seq.copy()
	    mod_seq = list(mod_seq)
	    while(len(mod_seq) < size_of_series):
	        random_element = random.choice([a for a in mod_seq if (mod_seq.index(a) != 0 and 
	                                                               mod_seq.index(a) != (len(mod_seq)-1))])
	        index_random = mod_seq.index(random_element)
	        avg_num = (mod_seq[index_random - 1] + mod_seq[index_random + 1])/2
	        mod_seq.insert(index_random, avg_num)  
	    while(len(mod_seq) > size_of_series):
	        del mod_seq[mod_seq.index(random.choice(mod_seq))]
	    return mod_seq   

	#DTW-Merge
	def dtw_aug_merge(seq1, seq2):
	    new_seq = []
	    _, warping_path = fastdtw.fastdtw(seq1, seq2)
	    L = len(warping_path)
	    random_num = int(np.random.normal(L/2, L/10))

	    if random_num < 0:
	    	random_num = 0
	    if random_num >= L:
	    	random_num = L - 1

	    new_seq.append([*seq1[:warping_path[random_num][0]], *seq2[warping_path[random_num][1]:]])
	    return new_seq

	#dataset augmentation
	def aug_dataset(self, samples_list, labels, sample_label_list, number_of_mul = 10):
	    augmented_dataset = samples_list.copy()
	    size_of_series = len(samples_list[0])
	    for i in range(len(samples_list)):
	        label_of_sample = labels[i]
	        for j in range(number_of_mul):
	            sample1 = random.choice([x for x in sample_label_list if x[1] == label_of_sample])[0]
	            sample2 = random.choice([x for x in sample_label_list if x[1] == label_of_sample])[0]

	            augmented_dataset.append(np.array(self.padding(self.dtw_aug_merge(sample1, sample2)[0], size_of_series = size_of_series)))
	            labels.append(label_of_sample)
	            
	    return augmented_dataset

	def normalizer(X):
	    X_norm = (X - np.mean(X)) / (np.std(X))
	    return X_norm 
	    
	#building ResNet model
	def build_model(n_feature_maps = 64, input_shape = 0, num_classes = 0):

		input_layer = tf.keras.layers.Input(input_shape)

		conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = tf.keras.layers.BatchNormalization()(conv_x)
		conv_x = tf.keras.layers.Activation('relu')(conv_x)

		conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = tf.keras.layers.BatchNormalization()(conv_y)
		conv_y = tf.keras.layers.Activation('relu')(conv_y)

		conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = tf.keras.layers.BatchNormalization()(conv_z)

		shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

		output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
		output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

#=========================================================================================================

		conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
		conv_x = tf.keras.layers.BatchNormalization()(conv_x)
		conv_x = tf.keras.layers.Activation('relu')(conv_x)

		conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
		conv_y = tf.keras.layers.BatchNormalization()(conv_y)
		conv_y = tf.keras.layers.Activation('relu')(conv_y)

		conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
		conv_z = tf.keras.layers.BatchNormalization()(conv_z)

		shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

		output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
		output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)


#=========================================================================================================

		conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
		conv_x = tf.keras.layers.BatchNormalization()(conv_x)
		conv_x = tf.keras.layers.Activation('relu')(conv_x)

		conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
		conv_y = tf.keras.layers.BatchNormalization()(conv_y)
		conv_y = tf.keras.layers.Activation('relu')(conv_y)

		conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
		conv_z = tf.keras.layers.BatchNormalization()(conv_z)

		shortcut_y = tf.keras.layers.BatchNormalization()(output_block_3)

		output_block_4 = tf.keras.layers.add([shortcut_y, conv_z])
		output_block_4 = tf.keras.layers.Activation('relu')(output_block_4)	 

		gap_layer = tf.keras.layers.GlobalAveragePooling1D()(output_block_4)

		output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

		model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

		return model
    
    
	def read_train_data(self, original_path, folder_name):
        
		folder_name_train = original_path + folder_name + '/' + folder_name + '_TRAIN.tsv'
    	    
		train_labels = []
		train_samples = []
    	    
    	    
		data_frame = pd.read_csv(folder_name_train, sep='\t', header=None)
		labels_of_dataset = data_frame[0]
		num_of_instances = len(data_frame)
		data_frame = data_frame.transpose()
    	    
		for i in range(num_of_instances):
			a = np.array(data_frame[i])
			if np.min(labels_of_dataset) == 0:
				train_labels.append(a[0])
			else:
				train_labels.append(a[0] - 1)
			train_samples.append(a[1:])
    
		train_samples = [x[np.logical_not(np.isnan(x))] for x in train_samples]
		lenghts = [len(x) for x in train_samples]
		mean_of_lenght = int(np.mean(lenghts))
    
    
		train_samples = [x[np.logical_not(np.isnan(x))] for x in train_samples]
		lenghts = [len(x) for x in train_samples]
		mean_of_lenght = int(np.mean(lenghts))
    
		num_classes = len(set(train_labels))
        

		if len(train_samples[0]) != mean_of_lenght:
			train_samples = [self.padding(x, mean_of_lenght) for x in train_samples]
    

        
		train_labels = (train_labels - min(train_labels))/(max(train_labels)-min(train_labels))*(num_classes-1)
		train_labels = list(train_labels)
        
		return train_samples, train_labels, num_classes
    
	def read_test_data(self, original_path, folder_name): 
    
		folder_name_test = original_path + folder_name + '/' + folder_name + '_TEST.tsv'
        
		test_labels = []
		test_samples = []


		data_frame_test = pd.read_csv(folder_name_test, sep='\t', header=None)
		num_of_instances = len(data_frame_test)
		data_frame_test = data_frame_test.transpose()
	    
		labels_of_dataset = data_frame_test[0]

		for i in range(num_of_instances):
			a = np.array(data_frame_test[i])
			if np.min(labels_of_dataset) == 0:
				test_labels.append(a[0])
			else:
				test_labels.append(a[0] - 1)
			test_samples.append(a[1:])

		test_samples = [x[np.logical_not(np.isnan(x))] for x in test_samples]

		test_samples_ind = [test_samples.index(x) for x in test_samples if len(x) <= 2]
		test_samples = [i for j, i in enumerate(test_samples) if j not in test_samples_ind]
		test_labels = [i for j, i in enumerate(test_labels) if j not in test_samples_ind]

		lenghts = [len(x) for x in test_samples]
		mean_of_lenght = int(np.mean(lenghts))

		if len(test_samples[0]) != mean_of_lenght:
			test_samples = [self.padding(x, mean_of_lenght) for x in test_samples]

		num_classes = len(set(test_labels))
        
		test_labels = (test_labels - min(test_labels))/(max(test_labels)-min(test_labels))*(num_classes-1)
      
	    #test_samples = [utils.normalizer(x) for x in test_samples]
        
		return test_samples, test_labels
