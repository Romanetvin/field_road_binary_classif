import os
# os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')

from datetime import datetime

import tensorflow.keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, BackupAndRestore, CSVLogger,LearningRateScheduler
from tensorflow.keras.optimizers import SGD

import config
from utils import *


def main():

	###################### Hyperparameters #####################
	# Setting up all the hyperparameters from config file
	date         = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

	SEED         = config.SEED

	BATCH_SIZE   = config.BATCH_SIZE
	EPOCHS       = config.EPOCHS
	LR           = config.LR
	GAMMA        = config.GAMMA
	STEP         = config.STEP
	VAL_SPLIT    = config.VAL_SPLIT
	AUGMENT      = config.AUGMENT

	ARCHITECTURE = config.ARCHITECTURE

	DATASET_PATH = config.DATASET_PATH
	TRAIN        = DATASET_PATH + 'train/'
	TEST         = DATASET_PATH + 'test/'
	IMG_SIZE     = config.IMG_SIZE

	MASKS_PATH   = config.MASKS_PATH

	RESUME_FILE  = config.RESUME_FILE

	CLASSES = ['fields', 'roads']

	############################################################


	####################### Callbacks ##########################

	# Create checkpoints folder if it doesn't exist
	model_checkpoint_folder = 'checkpoints/'
	if not os.path.isdir(model_checkpoint_folder):
		os.makedirs(model_checkpoint_folder, exist_ok=True)

	# Create checkpoint callback to save only if better val_dice_coef value
	model_checkpoint_name = model_checkpoint_folder + '_'.join([date, ARCHITECTURE, 'best_model.h5'])
	model_ckpt = ModelCheckpoint(
		filepath= model_checkpoint_name,
		save_weights_only=True,
		monitor='val_dice_coef',
		mode='max',
		save_best_only=True)

	# Create backup folder if it doesn't exist
	backup_folder = 'backup/'
	if not os.path.isdir(backup_folder):
		os.makedirs(backup_folder, exist_ok=True)

	# Create backup callback in case model gets interrupted during training
	backup_cb = BackupAndRestore(
		backup_dir =backup_folder,
		save_freq = 'epoch',
		delete_checkpoint = True
	)

	# Initialize csv log file
	csv_file = model_checkpoint_folder + 'logs.csv'

	# Create csv logger callback to save logs on a file to plot learning curves even if 
	# model gets interrupted
	csv_logger = CSVLogger(
		filename = csv_file,
		separator = ',',
		append = False)

	# Lr scheduler callback
	lr_scheduler = LearningRateScheduler(scheduler)

	#############################################################


	#################### Initialize datasets ####################
	train_gen = CustomDataGen(directory = TRAIN,
		masks_path = MASKS_PATH,
		batch_size = BATCH_SIZE,
		nb_classes = len(CLASSES)+1,
		subset = 'train',
		split = VAL_SPLIT,
		augment = AUGMENT)

	val_gen = CustomDataGen(directory = TRAIN,
		masks_path = MASKS_PATH,
		batch_size = BATCH_SIZE,
		nb_classes = len(CLASSES)+1,
		subset = 'val',
		split = VAL_SPLIT)

	#############################################################



	##################### Instanciate model #####################
	if ARCHITECTURE == 'transunet':
		model = models.transunet_2d(IMG_SIZE, filter_num=[64, 128, 256, 512], n_labels=len(CLASSES)+1, stack_num_down=2, stack_num_up=2,
						embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
						activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
						batch_norm=True, pool=False, unpool=False, name='transunet')

	elif ARCHITECTURE == 'attunet':
		model = models.att_unet_2d(IMG_SIZE, [64, 128, 256, 512], n_labels=len(CLASSES)+1,
						stack_num_down=2, stack_num_up=2,
						activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
						batch_norm=True, pool=False, unpool=False, name='attunet')

	elif ARCHITECTURE == 'resunet_a':
		model = models.resunet_a_2d(IMG_SIZE, [32, 64, 128, 256, 512, 1024], 
						dilation_num=[1, 3, 15, 31], 
						n_labels=len(CLASSES)+1, aspp_num_down=256, aspp_num_up=128, 
						activation='ReLU', output_activation='Softmax', 
						batch_norm=True, pool=False, unpool=False, name='resunet')


	elif ARCHITECTURE == 'unet3plus':
		model = models.unet_3plus_2d(IMG_SIZE, n_labels=len(CLASSES)+1, filter_num_down=[64, 128, 256, 512], 
						filter_num_skip='auto', filter_num_aggregate='auto', 
						stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Softmax',
						batch_norm=True, pool=False, unpool=False, deep_supervision=False, name='unet3plus')

	elif ARCHITECTURE == 'swinunet':
		model = models.swin_unet_2d(IMG_SIZE, filter_num_begin=64, n_labels=len(CLASSES)+1, depth=4, stack_num_down=2, stack_num_up=2, 
						patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
						output_activation='Softmax', shift_window=True, name='swin_unet')


	# Resume from file if there is any
	if RESUME_FILE is not None and len(RESUME_FILE)>0:
		print('[INFO] Resuming fro %s' % RESUME_FILE)
		model.load_weights(RESUME_FILE)

	# Compute alpha for each class. It corresponds to the inverse of the distribution of 
	# pixels for each class.
	alpha = compute_alpha(train_gen, len(CLASSES)+1)
	alpha[1] = alpha[1]*5
	alpha[2] = alpha[2]*5
	print(alpha)
	# Instanciate the loss function categorical focal loss
	# This loss is efficient on imbalanced dataset for semantic segmentation.
	LOSS = categorical_focal_loss(alpha = alpha, gamma = GAMMA)

	# Instanciate SGD optimizer as it was proven to be more stable on long training
	# compared to Adam
	OPT = SGD(learning_rate = LR, momentum = 0.9)


	# Compile model with loss, optimizer and metric
	model.compile(loss = LOSS, optimizer = OPT, metrics = dice_coef)

	##############################################################



	######################## Training ############################
	initial_epoch = 0

	model.fit(x = train_gen,
				batch_size = BATCH_SIZE,
				epochs = EPOCHS,
				steps_per_epoch = train_gen.__len__(),
				validation_data = val_gen,
				initial_epoch = initial_epoch,
				validation_steps = val_gen.__len__(),
				verbose = 1,
				callbacks = [csv_logger, backup_cb, model_ckpt, lr_scheduler]
		)
	#############################################################


	# Create results folder to store learning curves for each training
	if not os.path.isdir('results'):
		os.makedirs('results', exist_ok=True)

	learning_file = 'results/'+'_'.join([date, ARCHITECTURE, 'learning_curves.png'])
	# Plotting and saving learning curves on disk
	plot_learning_curves(csv_file=csv_file, destPath = learning_file)


	######################### Test model ########################
	# Instanciate test generator for final testing
	test_gen = CustomDataGen(directory = TEST,
		masks_path = MASKS_PATH,
		batch_size = BATCH_SIZE,
		nb_classes = len(CLASSES)+1,
		subset = 'test',
		to_categorical = False)

	# Evaluate model performance and returning dice score for test data
	test_dice_score = evaluate_model(model, test_gen)

	print('Test dice score : %.4f' % test_dice_score)
	#############################################################


	################### Hyperparameters saving ##################
	hyperparameters_file = 'hyperparameters.csv'

	save_hyperparameters(hyperparameters_file, date, test_dice_score, learning_file, model_checkpoint_name.split('/')[-1])
	#############################################################




if __name__ == '__main__':
	main()