import os
import sys
import config
import glob
import csv
import PIL
from PIL import Image
import math
import numpy as np
import random
from resizeimage import resizeimage
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import load_img, img_to_array, to_categorical


import albumentations as A

sys.path.insert(1, 'keras_unet_collection/')
from keras_unet_collection import models

np.random.seed(42)


def scheduler(epoch, lr):
	"""
	Learning rate scheduler

	Args:
		epoch : epoch number
		lr    : learning rate

	Returns:
		Updated learning rate

	"""
	if epoch == config.STEP:
		return lr / 10.
	else:
		return lr

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def compute_alpha(dataset, nb_classes):
	"""
	Compute alpha parameter for categorical_focal_loss
	
	Args:
		dataset    : generator from which to compute alpha
		nb_classes : number of total classes

	Returns:
		List of inverse of per class pixel distribution
	"""
	labels_count = [0]*nb_classes
	for i in range(dataset.__len__()):
		imgs, masks = dataset.__getitem__(i)
		masks = np.argmax(masks, axis = -1)
		uniques, counts = np.unique(masks, return_counts = True)
		uniques = list(uniques.astype(int))
		for u in uniques:
			labels_count[u] = labels_count[u] + counts[uniques.index(u)]
	return 1 / (labels_count / np.sum(labels_count))

# Keras metric
def dice_coef(y_true, y_pred, smooth=100):
	"""
	Keras metric for dice coefficient
	
	Args:
		y_true : ground-truth label mask 
		y_pred : predited label mask
		smooth : smooth value to avoid Divide by zero error
	Returns:
		Dice score

	"""
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return dice

# Numpy metric
def dice(im1, im2):
	"""
	Computes the Dice coefficient, a measure of set similarity.
	Parameters
	----------
	im1 : array-like, bool
	Any array of arbitrary size. If not boolean, will be converted.
	im2 : array-like, bool
		Any other array of identical size. If not boolean, will be converted.
	Returns
	-------
	dice : float
		Dice coefficient as a float on range [0,1].
		Maximum similarity = 1
		No similarity = 0

	Notes
	-----
	The order of inputs for `dice` is irrelevant. The result will be
	identical if `im1` and `im2` are switched.
	"""
	im1 = np.asarray(im1).astype(bool)
	im2 = np.asarray(im2).astype(bool)

	if im1.shape != im2.shape:
		raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

	# Compute Dice coefficient
	intersection = np.logical_and(im1, im2)

	return 2. * intersection.sum() / (im1.sum() + im2.sum())

def save_hyperparameters(file, date, test_score, learning_curves_file, best_model_file):
	"""
	Function to save hyperparameters to a .csv file to have a trace of every training results

	Args:
		file                 : filename to save the hyperparameters
		date                 : date when model training started
		test_score           : dice test score after the training on the test data
		learning_curves_file : filename of the learning curves
		best_model_file      : filename of the best model

	"""

	field_names = ['date', 'test_score', 'seed', 'batch_size',
				   'epochs', 'lr', 'gamma', 'step', 'val_split',
				   'img_size', 'resume_file','augment',
				   'learning_curves_file', 'best_model_file']
	to_write_header = False

	hyperparameters_dict = {
		'date' : date,
		'test_score': test_score,
		'seed': config.SEED,
		'batch_size': config.BATCH_SIZE,
		'epochs': config.EPOCHS,
		'lr':config.LR,
		'gamma':config.GAMMA,
		'step':config.STEP,
		'val_split':config.VAL_SPLIT,
		'augment':config.AUGMENT,
		'img_size': str(config.IMG_SIZE),
		'resume_file':config.RESUME_FILE,
		'learning_curves_file' : learning_curves_file,
		'best_model_file' : best_model_file
	}

	if not os.path.isfile(file):
		to_write_header = True


	with open(file, 'a') as f:
		dictwriter = csv.DictWriter(f, fieldnames = field_names)
		if to_write_header:
			dictwriter.writeheader()
		dictwriter.writerow(hyperparameters_dict)

		f.close()

def plot_learning_curves(csv_file, destPath):
	"""
	Plot and save learning curves of a training

	Args:
		csv_file : log file where the data was saved during training
		destPath : filename to where the learning curves plot are saved

	"""
	data = {'loss':[], 'val_loss':[], 'dice_coef':[], 'val_dice_coef':[]}
	with open(csv_file, 'r') as f:
		csvreader = csv.DictReader(f, delimiter=',')
		# Skip header
		for row in csvreader:
			data['loss'] = data['loss']+[float(row['loss'])]
			data['val_loss'] = data['val_loss']+[float(row['val_loss'])]
			data['dice_coef'] = data['dice_coef']+[float(row['dice_coef'])]
			data['val_dice_coef'] = data['val_dice_coef']+[float(row['val_dice_coef'])]

		f.close()

	fig, (ax1, ax2) = plt.subplots(2, 1)

	loss1 = ax1.plot([i+1 for i in range(len(data['loss']))], data['loss'], label='loss')
	loss2 = ax1.plot([i+1 for i in range(len(data['val_loss']))], data['val_loss'], label='val loss')

	ax1.legend(loc='upper right')

	ax1.set_title('Loss curves')
	ax1.set_ylabel('Loss value')


	ax2.plot([i+1 for i in range(len(data['dice_coef']))], data['dice_coef'], label='dice')
	ax2.plot([i+1 for i in range(len(data['val_dice_coef']))], data['val_dice_coef'], label='val dice')

	ax2.legend(loc='lower right')

	ax2.set_title('Dice scores')
	ax2.set_ylabel('Dice value')
	ax2.set_ylim([0, 1])

	fig.suptitle('Learning curves')

	fig.savefig(destPath)

def evaluate_model(model, test_gen):
	"""
	Evaluate model performance on semantic segmentation

	Args:
		model    : model instance
		test_gen : test generator 

	Returns:
		Dice score on test data

	"""
	y_true, y_pred = [], []
	for batch in test_gen:
		X, y = batch
		pred = np.argmax(model(X), axis = -1)

		y_true.extend(y)
		y_pred.extend(pred)

	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	

	return dice(y_true, y_pred)

class CustomDataGen(tf.keras.utils.Sequence):
	"""
	Custom data generator for semantic segmentation

	"""
	def __init__(self, directory, masks_path, batch_size, nb_classes, subset = 'train', shuffle=True, split = 0.8, to_categorical = True, augment = False, return_filenames = False):
		"""
		Initialize generator

		Args:
			directory      : root directory for images
			masks_path     : path of all the masks
			batch_size     : batch size
			nb_classes     : number of classes in the segmentation task
			subset         : which subset to instanciate ('train' or 'val') to split the data
			shuffle        : whether to shuffle or not the dataset at the beginning and after each epoch
			split          : split size
			to_categorical : whether to return the labels as values or one-hot encoded vectors
			augment        : whether to apply online data augmentation during training

		"""
		self.directory = directory
		self.masks_path = masks_path
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.shuffle = shuffle
		self.to_categorical = to_categorical
		self.subset = subset
		self.augment = augment
		self.return_filenames = return_filenames

		# Create tuples with (img, mask)
		self.dataset = self.create_image_mask_tuple()

		# Shuffle data 
		if self.shuffle:
			random.seed(63)
			self.shuffle_data()

		# Split dataset depending on subset
		if subset == 'train':
			self.dataset = self.dataset[:int(len(self.dataset) * (1-split))]

		elif subset == 'val':
			self.dataset = self.dataset[-int(len(self.dataset) * split):]


		print('Number of images in %s subset : %d' % (self.subset, len(self.dataset)))

		self.n = len(self.dataset)

	def on_epoch_end(self):
		"""
		Action to do at the end of an epoch
		"""
		self.shuffle_data()

	def create_image_mask_tuple(self):
		"""
		Get the (img, mask) pair for each file 

		Returns:
			A list of all tuples with (img, mask) pairs

		"""
		list_folders = list(sorted(glob.glob(self.directory+'*/')))
		if self.masks_path is not None:
			list_masks = list(sorted(glob.glob(self.masks_path+'*.png')))

		list_data = []

		for folder in list_folders:
			list_imgs = list(sorted(glob.glob(folder+'*')))

			for img_path in list_imgs:
				if self.masks_path is not None:
					list_data.append((img_path, [elt for elt in list_masks if elt.split('/')[-1][:-4] in img_path][0]))
				else:
					list_data.append(img_path)


		return list_data

	def shuffle_data(self):
		"""
		Shuffle data
		"""
		random.shuffle(self.dataset)

	def squarify(self, pil_image, interpolation = Image.LANCZOS, bg_color = (0,0,0)):
		"""
		Make the image square

		Args:
			pil_image : PIL Image

		Returns:
			a squared tensor and resized to IMG_SIZE in the config file

		"""
		def squarify_fixed(pil_image, img_size, interpolation, bg_color):
			# pil_image = pil_image.convert('RGBA').convert('RGB')

			img = resizeimage.resize_contain(
				image = pil_image,
				size= img_size,
				resample=interpolation,
				bg_color=bg_color)
			img = img.convert('RGB')
			return img_to_array(img)

		return squarify_fixed(pil_image = pil_image, img_size = config.IMG_SIZE,
			interpolation =interpolation, bg_color=bg_color)

	def convert_png2label(self, mask):
		"""
		Convert a png mask into a label mask with (0,1,2) values
	
		Args:
			mask : numpy array of mask

		Returns:
			Label mask with either (0,1,2) values or one-hot encoded vector
		"""
		bg_color = np.array([0, 113, 188])
		field = np.array([216, 82, 24])
		road = np.array([236, 176, 31])


		label = np.zeros((mask.shape[0], mask.shape[1]))

		label[np.all(mask == bg_color, axis=-1)] = 0

		label[np.all(mask == field, axis=-1)] = 1

		label[np.all(mask == road, axis=-1)] = 2

		if self.to_categorical:
			return to_categorical(label, num_classes = self.nb_classes)
		else:
			return label

	def augment_(self, img, mask):
		"""
		Apply same data augmentation on image and mask

		Args:
			img  : numpy array of image
			mask : numpy array of mask

		"""
		# Set augmentation transformations to apply
		aug = A.Compose([
				A.HorizontalFlip(p=0.3),
				A.VerticalFlip(p=0.3),
    			A.RandomRotate90(p=0.3),
    			# A.RandomSizedCrop(min_max_height=(50, 200), 
				# 		height=config.IMG_SIZE[0], width=config.IMG_SIZE[1], p=0.3),
			    # A.RandomBrightnessContrast(p=0.3),    
			    # A.RandomGamma(p=0.3)
			])


		augmented = aug(image = img, mask=mask)
		return augmented['image'], augmented['mask']

	def __get_data(self, batches):
		"""
		Load data for current batch

		Args:
			batches : list of tuple (img, mask) pairs to load

		Returns:
			Numpy arrays for the current batch imgs and masks

		"""
		img_batch, mask_batch = [], []
		for sample in batches:
			if self.masks_path is not None:
				img_path, mask_path = sample
			else:
				img_path = sample

			# Load pil image and apply squarify to make image square
			img = self.squarify(load_img(img_path, color_mode = 'rgb'),
				interpolation = Image.LANCZOS)

			img_batch.append(img)

			if self.masks_path is not None:

				# Load mask as pil image, squarify and replace bg_color by mask bg color
				# (0, 113, 188) corresponds to the BG color on masks
				mask = self.squarify(load_img(mask_path, color_mode = 'rgb'),
					interpolation = Image.NEAREST, bg_color = (0 ,113, 188))

				mask = self.convert_png2label(mask)

				if self.augment:
					img, mask = self.augment_(img,mask)
				mask_batch.append(mask)


		if self.masks_path is not None:
			return np.array(img_batch), np.array(mask_batch)
		else:
			return np.array(img_batch)

    
	def __getitem__(self, index):
		"""
		Get batch from index number
		
		Args:
			index : index number during epoch
		
		Returns:
			Imgs batch and masks batch
		"""

		batches = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
		if self.masks_path is not None:
			X, y = self.__get_data(batches)
			return X, y
		else:
			X = self.__get_data(batches)

			if self.return_filenames:
				return X, batches
			
			return X
    
	def __len__(self):
		if self.n % self.batch_size == 0:
			return self.n // self.batch_size
		else:
			return (self.n // self.batch_size) + 1

