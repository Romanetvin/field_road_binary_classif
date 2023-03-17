# Hyperparameters
SEED         = 631

BATCH_SIZE=8
EPOCHS=200
LR=0.03
GAMMA=2.
STEP=100
VAL_SPLIT=0.2
AUGMENT=False
# ['attunet', 'unet3plus', 'resunet_a','transunet', 'swinunet']
ARCHITECTURE='resunet_a'

DATASET_PATH = '/home/vinnie/Documents/Python/Bilberry/dataset/'
MASKS_PATH   = DATASET_PATH + 'masks/'
IMG_SIZE     = (256,256,3)
RESUME_FILE=None
# RESUME_FILE = '16-03-2023_20:53:19_attunet_best_model.h5'
# RESUME_FILE = 'checkpoints/'+RESUME_FILE
