import os
import sys
from itertools import product

def update_line(line, new_value):
	old_value = line.split('=')[-1].strip('\n')
	new_line = line.replace(old_value, new_value)
	return new_line



##################### Parameters ########################
batch_sizes = ['4']
epochs      = ['200']
lrs         = ['0.003']
gammas      = ['2.']
augments    = ['True']
architectures = ['\'attunet\'', '\'unet3plus\'', '\'resunet_a\'','\'transunet\'']
#########################################################

# config file to modify
config_file = 'config.py'

# Loop over the product of all parameters lists
for batch_size, epoch, lr, gamma, augment, architecture in 
	product(batch_sizes, epochs, lrs, gammas, augments, architectures):

	# Read config file
	with open(config_file, 'r') as cf:
		lines = cf.readlines()

	# Variables to try the model training on
	variables = [batch_size, epoch, lr, gamma, augment, architecture]

	# Transforming the variables name into string
	var_to_str = [
					f'{batch_size=}'.split('=')[0],
					f'{epoch=}'.split('=')[0],
					f'{lr=}'.split('=')[0],
					f'{gamma=}'.split('=')[0],
					f'{augment=}'.split('=')[0],
					f'{architecture=}'.split('=')[0],
					]

	# Loop over each line in file
	for i in range(len(lines)):
		# Checking if a str variable is in the line and returning a list of booleans for each variable
		var_list = list(elt in lines[i].lower() for elt in var_to_str)
		# if true in the list
		if any(var_list):
			# Update line with corresponding variable value
			lines[i] = update_line(lines[i], variables[[i for i,x in enumerate(var_list) if x][0]])
			
	# Rewrite all the lines to the file
	with open(config_file, 'w') as cf:
		for line in lines:
			cf.write(line)
		cf.close()

	# Launch training
	os.system('python train.py')