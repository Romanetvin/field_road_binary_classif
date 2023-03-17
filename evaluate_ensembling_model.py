import os
os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import sys

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import load_img, img_to_array, to_categorical

from sklearn.metrics import cohen_kappa_score, confusion_matrix

import cv2

sys.path.insert(1, 'keras_unet_collection/')
from keras_unet_collection import models

import config
from utils import CustomDataGen, dice

CLASSES = ['fields', 'roads']


def flip_aug(batch_img):
	"""
	Apply horizontal, vertical and h+v flips on testing images
	
	Args:
		batch_img : current batch of images of shape (N, h, w, c), N : batch size, h : height, w : width, c : channel

	Returns:
		Numpy array of flipped images batch of shape (N, 4, h, w, c), 4 being the number of inference augmentations
	"""
	flip_batch_img = []
	for i in range(batch_img.shape[0]):
		flip_img = []
		flip_img.append(batch_img[i])
		# Flip horizontaly
		flip_img.append(np.fliplr(batch_img[i]))
		# Flip verticaly
		flip_img.append(np.flipud(batch_img[i]))
		# Flip horizontaly and verticaly
		flip_img.append(np.flipud(np.fliplr(batch_img[i])))

		flip_batch_img.append(flip_img)

	return np.array(flip_batch_img)

def unflip_aug(batch_img):
	"""
	Unflip masks predictions

	Args:
		batch_img : current batch of masks predictions of shape (N, 4, h, w, c), N : batch size, 4: nb transformations, h : height, w : width, c : channel

	Returns:
		Numpy array of unflipped masks predictions of shape (N, 4, h, w, c)

	"""
	unflip_batch_img =[]
	for i in range(batch_img.shape[0]):
		unflip_img = []
		unflip_img.append(batch_img[i][0])
		# Flip horizontaly
		unflip_img.append(np.fliplr(batch_img[i][1]))
		# Flip verticaly
		unflip_img.append(np.flipud(batch_img[i][2]))
		# Flip horizontaly and verticaly
		unflip_img.append(np.flipud(np.fliplr(batch_img[i][3])))

		unflip_batch_img.append(unflip_img)

	return np.array(unflip_batch_img)

def convert_mask2binary(masks):
	"""
	Convert masks (gt or preds) to binary list for binary classification task

	Args:
		masks : numpy array of masks labels with (0,1,2) values
	

	Returns:
		Numpy array of new labels (0,1) for binary classification
	"""
	# fields = 0, rodas = 1
	labels = []
	for mask in masks:
		# For each mask, count number of each value
		uniques, counts = np.unique(mask, return_counts=True)
		# If there are both field (1) and road (2) values
		if 1 in uniques and 2 in uniques:
			# Assign the label for the highest counts number
			if counts[uniques[1]] > counts[uniques[2]]:
				labels.append(0)
			elif counts[uniques[1]] < counts[uniques[2]]:
				labels.append(1)
		else:
			labels.append(0 if 1 in uniques else 1)

	return np.array(labels)

def convert_label2png(mask):
	# RGB colors
	# bg_color = np.array([0, 113, 188])
	# field = np.array([216, 82, 24])
	# road = np.array([236, 176, 31])

	# BGR colors
	bg_color = np.array([188, 113, 0])
	field = np.array([24, 82, 216])
	road = np.array([31, 176, 236])


	label = np.zeros((mask.shape[0], mask.shape[1], 3))

	label[mask == 0] = bg_color
	label[mask == 1] = field
	label[mask == 2] = road

	return label

def load_model_weights(weights_path, architecture):
	"""
	Load model weights depending on the architecture

	Args:
		weights_path : path to the weights to load
		architecture : string of architecture to load


	"""
	if architecture == 'transunet':

		model = models.transunet_2d(config.IMG_SIZE, filter_num=[64, 128, 256, 512], n_labels=len(CLASSES)+1, stack_num_down=2, stack_num_up=2,
	                                embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
	                                activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
	                                batch_norm=True, pool=False, unpool=False, name='transunet')

	elif architecture == 'attunet':
		model = models.att_unet_2d(config.IMG_SIZE, [64, 128, 256, 512], n_labels=len(CLASSES)+1,
				stack_num_down=2, stack_num_up=2,
				activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
				batch_norm=True, pool=False, unpool=False, name='attunet')

	elif architecture == 'resunet_a':
		model = models.resunet_a_2d(config.IMG_SIZE, [32, 64, 128, 256, 512, 1024], 
                            dilation_num=[1, 3, 15, 31], 
                            n_labels=len(CLASSES)+1, aspp_num_down=256, aspp_num_up=128, 
                            activation='ReLU', output_activation='Softmax', 
                            batch_norm=True, pool=False, unpool=False, name='resunet')

	elif architecture == 'unet3plus':
		model = models.unet_3plus_2d(config.IMG_SIZE, n_labels=len(CLASSES)+1, filter_num_down=[64, 128, 256, 512], 
                             filter_num_skip='auto', filter_num_aggregate='auto', 
                             stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Softmax',
                             batch_norm=True, pool=False, unpool=False, deep_supervision=False, name='unet3plus')

	elif architecture == 'swinunet':
		model = models.swin_unet_2d(config.IMG_SIZE, filter_num_begin=64, n_labels=len(CLASSES)+1, depth=4, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swin_unet')

	# Load model weights 
	model.load_weights(weights_path)

	return model

def evaluate_ensembling(list_weights, list_architectures, gen_args, save_masks = False):

	"""
	Evaluate on test data or other data, possibility to save predicted masks

	Args: 
		list_weights     : list of weights file to load the models from
		list_archectures : list of architectures name to load

	"""

	models = [load_model_weights(weights, architecture) for weights, architecture in zip(list_weights, list_architectures)]
	
	test_gen = CustomDataGen(directory = ''.join((config.DATASET_PATH, gen_args['subset'])),
		batch_size = config.BATCH_SIZE,
		nb_classes = len(CLASSES)+1,
		shuffle = False,
		to_categorical = False, 
		**gen_args)


	dice_scores = []	
	y_true, y_pred = [], []

	imgs_to_save = []
	masks_to_save = []
	filenames_to_save = []


	for i in range(test_gen.__len__()):
		if gen_args['masks_path'] is not None:
			imgs, masks = test_gen.__getitem__(i)
		else:
			if gen_args['return_filenames']:
				imgs, filenames = test_gen.__getitem__(i)
				filenames_to_save.extend(filenames)
			else:
				imgs = test_gen.__getitem__(i)
		final_preds = []
		flipped_imgs = flip_aug(imgs)
		for flipped_img in flipped_imgs:

			pred = [model.predict(flipped_img) for model in models]

			unflipped_pred = unflip_aug(np.array(pred))

			# Average on probabilities
			avg_on_flip = np.mean(unflipped_pred, axis=1)
			avg_on_model = np.mean(avg_on_flip, axis = 0)
			final_preds.append(avg_on_model)
			

		final_preds = np.argmax(np.array(final_preds), axis=-1)
		imgs_to_save.extend(imgs)
		masks_to_save.extend(final_preds)
		

		if gen_args['masks_path'] is not None:
			y_true.extend(convert_mask2binary(masks))
			y_pred.extend(convert_mask2binary(final_preds))


			dice_scores.append(dice(masks, final_preds))


	if gen_args['masks_path'] is not None:
		final_dice = np.mean(dice_scores)

		print(y_true)
		print(y_pred)

		final_cohen_kappa = cohen_kappa_score(y_true, y_pred)

		print('Test dice score on ensembling : %.4f' % final_dice)

		print('Test cohen kappa score : %.4f' % final_cohen_kappa)



	if save_masks:
		dest_path = 'predicted_masks'
		if not os.path.isdir(dest_path):
			os.makedirs(dest_path, exist_ok=True)

		masks_to_save = np.array(masks_to_save)

		binary_preds = convert_mask2binary(masks_to_save)

		print('Binary output for each image : %s' %binary_preds)
		for img, mask, filename, b_pred in zip(imgs_to_save, masks_to_save, filenames_to_save, binary_preds):
			png_mask = convert_label2png(mask)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


			name = '/'.join((dest_path, 'predicted_'+CLASSES[b_pred]+'_gt_'+filename.split('/')[-1].split('.')[0]+'.png'))
			h_stack = np.hstack((img, png_mask)).astype(int)
			
			cv2.imwrite(name, h_stack)




if __name__ == '__main__':

	

	#############################
	# Parameters to save masks on images that don't have gt masks

	# 'subset' : Can be either ['test/', 'extra_test/'] or even another folder added to the dataset
	gen_args = {
		'subset'            : 'extra_test/',
		'masks_path'        : None,
		'return_filenames'  : True,
	}
	SAVE_MASKS = True
	#############################

	#############################
	# # Parameters to get test dice score and cohen kappa score on images that havet gt masks
	# # and to not save masks
	# # 'subset' : Can be either ['test/', 'extra_test/'] or even another folder added to the dataset
	# gen_args = {
	# 	'subset'            : 'test/',
	# 	'masks_path'        : config.MASKS_PATH,
	# 	'return_filenames'  : False,
	# }
	# SAVE_MASKS = False
	#############################






	list_selected_models = [
		'16-03-2023_17:31:28_resunet_a_best_model.h5',
		'17-03-2023_07:31:49_unet3plus_best_model.h5',
		'17-03-2023_14:48:01_resunet_a_best_model.h5',
		'17-03-2023_15:30:28_attunet_best_model.h5'
	]

	list_selected_models = ['checkpoints/' +elt for elt in list_selected_models]


	list_architectures = ['_'.join(elt.split('_')[2:-2]) for elt in list_selected_models]
	print(list_architectures)

	evaluate_ensembling(list_weights = list_selected_models, list_architectures = list_architectures, gen_args = gen_args, save_masks = SAVE_MASKS)