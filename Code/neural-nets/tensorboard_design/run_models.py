"""
usage: run_models.py 
------------------------------------------------------------
                [-h] [--layers LAYERS [LAYERS ...]]
                [--width WIDTH [WIDTH ...]]
                [--lrate LRATE [LRATE ...]]
                [--hacts {sigmoid,tanh,relu,softmax,softsign,elu} [{sigmoid,tanh,relu,softmax,softsign,elu} ...]]
                [--yacts {sigmoid,tanh,relu,softmax,softsign,elu} [{sigmoid,tanh,relu,softmax,softsign,elu} ...]]
                [-o {Adam,ADAgrad,ADAdelta,GD,PAD,RMSprop} [{Adam,ADAgrad,ADAdelta,GD,PAD,RMSprop} ...]]
                [--loss {cosine,log,hinge,mse,smax_ce,sig_ce,cross_entropy} [{cosine,log,hinge,mse,smax_ce,sig_ce,cross_entropy} ...]]
                [-i {norm,uni,trunc_norm,zeroes,constant,ones} [{norm,uni,trunc_norm,zeroes,constant,ones} ...]]
                [--init_vars INIT_VARS [INIT_VARS ...]] [-e EPOCHS]
                [-b BATCH_SIZE] [-d DATASIZE]
                [-r {basic,grid_search,random_search,datasizes,timetaken}]
                [--runs RUNS] [--popt_iter POPT_ITER]
                p_txt_path rand_path c_txt_dir results_string
                {C1,C2,P1,P2}

optional arguments:
  -h, --help            show this help message and exit

Directories:
------------------------------------------------------------
  Arguments for passing file/directory paths

  p_txt_path            <R> Plain texts bin DIRECT filepath
  rand_path             <R> Eithe the random texts bin OR key bin DIRECT
                        filepath
  c_txt_dir             <R> Cipher text directory
  results_string        <R> Results folder name

Run time properties:
------------------------------------------------------------
  Arguments for how to run the nnet

  {C1,C2,P1,P2}         <R> Which task (C1,C2,P1,P2)
  -e EPOCHS, --epochs EPOCHS
                        Choose max. number of epochs (default 100)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Choose batch sizes (default 20k)
  -d DATASIZE, --datasize DATASIZE
                        No. of data points to use (0->x), uses all if not
                        provided
  -r {basic,grid_search,random_search,datasizes,timetaken}, --run_type {basic,grid_search,random_search,datasizes,timetaken}
                        Type of run (basic, grid_search, random_search,
                        datasizes, timetaken) (default basic)
  --runs RUNS           Number of reliability runs (default 1)
  --popt_iter POPT_ITER
                        how many popt random iterations to do

Network properties:
------------------------------------------------------------
  Arguments for how the nnet is built

  --layers LAYERS [LAYERS ...]
                        max layers (for param optimiseenter two numbers)
  --width WIDTH [WIDTH ...]
                        widths of layers. for popt, provide 2 entries: start
                        end
  --lrate LRATE [LRATE ...]
                        learning rate. for popt, provide 2 entries: start end
  --hacts {sigmoid,tanh,relu,softmax,softsign,elu} [{sigmoid,tanh,relu,softmax,softsign,elu} ...]
                        choose h act. funcs. of net. (default sigmoid)
  --yacts {sigmoid,tanh,relu,softmax,softsign,elu} [{sigmoid,tanh,relu,softmax,softsign,elu} ...]
                        choose y act. funcs. of net. (default sigmoid)
  -o {Adam,ADAgrad,ADAdelta,GD,PAD,RMSprop} [{Adam,ADAgrad,ADAdelta,GD,PAD,RMSprop} ...], --optimiser {Adam,ADAgrad,ADAdelta,GD,PAD,RMSprop} [{Adam,ADAgrad,ADAdelta,GD,PAD,RMSprop} ...]
                        choose optimiser
  --loss {cosine,log,hinge,mse,smax_ce,sig_ce,cross_entropy} [{cosine,log,hinge,mse,smax_ce,sig_ce,cross_entropy} ...]
                        choose loss func.
  -i {norm,uni,trunc_norm,zeroes,constant,ones} [{norm,uni,trunc_norm,zeroes,constant,ones} ...], --inits {norm,uni,trunc_norm,zeroes,constant,ones} [{norm,uni,trunc_norm,zeroes,constant,ones} ...]
                        type of weights init
  --init_vars INIT_VARS [INIT_VARS ...]
                        Weights, bias inital values

"""

# Global module import
################################################################################

import os, tensorflow as tf, numpy as np
from sys import stdout as console
from random import random as rand, randint, uniform as randu
from datetime import datetime as dt

# Misc Functions
################################################################################

def working_directory_check():
	
	"""
	working_directory_check
		Check if the script is being run from correct directory
		Not actually required (can pass relative paths to args now)
		e.g. ../../data-gen/c-text-data/file.txt
		My folder structure is as so:
		- Main folder
		-	neural-nets folder
		-	data folder
	"""
	
	if os.getcwd() != '/home/dijksterhuis/SymNets/':
		os.chdir('/home/dijksterhuis/SymNets/')

def s(string):
	
	"""
	s
		Convert something to string type, but avoid having to write 3 characters in code
		I don't like wide lines, I get confused.
	"""
	
	return str(string)

# Data Transforms + Preperation
################################################################################

def get_file(filepath,max_size=0):
	
	"""
	get_file
		Get the txt data file from filepath passed
		if no max_size provided (defaults to 0), get all the data
	"""
	
	with open(filepath) as f:
		if max_size > 0:
			bits_all = [get_file_subfunc(next(f)) for x in range(max_size)]
		else:
			bits_all = [get_file_subfunc(line) for line in f.readlines()]
	return bits_all

def get_file_subfunc(a):

	"""
	get_file_subfunc
		this is to do with the way I generated casaer cipher data
		casaer cipher had been stored as python bin type (i.e. '0b0001010')
		whereas the rest are pure bit strings e.g. 00010101010100010
	"""

	if 'b' in a:
		x = [int(i) for i in a.rstrip('\n').rsplit('b',1)[1]]
	else:
		x = [int(i) for i in a.rstrip('\n')]
	return x

def to_byte_fromsingleint(integer):
	
	"""	
	to_byte_fromsingleint
		for convertign an integer value to binary value
		used for doing PM2 problem (see below - converts int to binary)
	"""
	
	split_byte = bin(integer).split('b')
	byte_pad = ''.join(['0' for i in range(0,8 - len(split_byte[1]))])
	padded_byte = byte_pad + split_byte[1]
	return padded_byte

def gen_C1_inputs(p_txts,c_txts,r_txts):
	
	"""
	gen_C1_inputs
		Provided p, c and r txts, randomly generate the C1 problem data set
		randomly decide which rows are random / cipher concatenations
		50/50 choice between each
		TODO write a double row function for triplet/hinged loss as discussed
	"""
	
	class_list, in_data = list(), list()
	for i in range(0,len(c_txts)):
		choice = randint(0,1)
		if choice is 1:
			in_data.append(p_txts[i] + c_txts[i])
		else:
			in_data.append(p_txts[i] + r_txts[i])
		class_list.append([choice])
	return in_data, class_list

def gen_C2_inputs(p_txts,c_txts):
	
	"""
	gen_C2_inputs
		Provided ptxts and ctxts, randomly generate C2 data sets
		As C1, but added complciation of choosing from 0-4 key values
	"""
	
	keys = list(c_txts.keys())
	class_1_file = keys[0]
	class_list, in_data = list(), list()
	for i in range( 0 , len(c_txts[ class_1_file ] ) ):
		choice = randint( 0 , 1 )
		if choice is 1:
			in_data.append( p_txts[i] + c_txts[ class_1_file ][i] )
		else:
			key_choice = randint( keys[ 1 ] , keys[ len(keys)-1 ] )
			in_data.append( p_txts[ i ] + c_txts[ key_choice ][ i ])
		class_list.append( [ choice ] )
	return in_data, class_list

def gen_PM2_inputs(p_txts,c_txts,c_keys):
	
	"""
	gen_PM2_inputs
		Provided ptxts and ctxts, randomly generate PM2 data sets
	"""
		
	class_list, in_data = list(), list()

	for i in range(len(p_txts)):
		
		choice = randint( min(c_txts.keys()) , max(c_txts.keys()) )
		in_data.append( p_txts[i] + c_txts[choice][i] )
		class_list.append( c_keys[choice  - min(c_txts.keys()) ] )
		
	return in_data, class_list

def train_valid_test_data(inputs,outputs,data_size,splits=[0.6,0.2,0.2]):
	
	"""
	train_valid_test_data
		Create Train, Valid and Test data sets from provided inputs/outputs (lists)
	"""

	train_len, valid_len, test_len = int(data_size * splits[0]), int(data_size * splits[1]), int(data_size * splits[2])
	
	train_dim = [ 0 , train_len ]
	valid_dim = [ train_len+1 , train_len + valid_len ]
	test_dim = [ valid_len+1 , valid_len + test_len ]

	inputs_train = inputs[ train_dim[0] : train_dim[1] ]
	inputs_valid = inputs[ valid_dim[0] : valid_dim[1] ]
	inputs_test = inputs[ test_dim[0] : test_dim[1] ]

	outputs_train = outputs[ train_dim[0] : train_dim[1] ]
	outputs_valid = outputs[ valid_dim[0] : valid_dim[1] ]
	outputs_test = outputs[ test_dim[0] : test_dim[1] ]

	print( 'to train : ' + s(len(outputs_train)) + ' ~ ' + s( int( ( len( outputs_train ) * 100 ) / len( inputs ) )) + '%' + 'of data')
	print( 'to validate : ' + s(len(outputs_valid)) + ' ~ ' + s( int( ( len( outputs_valid ) * 100 ) / len( inputs ) )) + '%' + 'of data')
	print( 'to test : ' + s(len(outputs_test)) + ' ~ ' + s( int( ( len( outputs_test ) * 100 ) / len( inputs ) )) + '%' + 'of data')

	return inputs_train, outputs_train, inputs_valid, outputs_valid, inputs_test, outputs_test

def get_and_prepare_data( problem , ptxt_path , c_txt_dir , rand_path , data_size = 0):
	
	"""
	get_and_prepare_data
		Combine several of the above functions and get the right data
		N.B. DO NOT USE FOR CASAER DATA
	"""
	
	if problem == 'C1':
		c_files = list(os.walk(c_txt_dir))[0][2]
		ctxt_path = c_txt_dir + ''.join([x for x in c_files if '1' in x and 'bin' in x and '.txt' in x])
		p_txts = get_file( ptxt_path , max_size = data_size )
		c_txts = get_file( ctxt_path , max_size = data_size )
		r_txts = get_file( rand_path , max_size = data_size )
		in_data, class_list = gen_C1_inputs(p_txts,c_txts,r_txts)

	elif problem == 'C2':
		p_txts = get_file( ptxt_path , max_size = data_size)
		c_files = [k for i,j,k in os.walk(c_txt_dir)][0]
		c_txts = { int(a_file.split('_')[1]) : get_file( c_txt_dir + a_file , max_size = data_size) for a_file in c_files if 'bin' in a_file}
		in_data, class_list = gen_C2_inputs(p_txts,c_txts)
		
	elif problem == 'P1':
		c_files = list(os.walk(c_txt_dir))[0][2]
		ctxt_path = c_txt_dir + ''.join([x for x in c_files if '_0_' in x and 'bin' in x and '.txt' in x])
		in_data , class_list = get_file( ptxt_path , max_size = data_size ) , get_file( ctxt_path  , max_size = data_size )

	elif problem == 'P2':
		p_txts = get_file( ptxt_path , max_size = data_size)
		c_files = list(os.walk(c_txt_dir))[0][2]
		c_txts = { int(a_file.split('_')[1]) : get_file( c_txt_dir + a_file , max_size = data_size  ) for a_file in c_files if 'bin' in a_file}
		c_keys = get_file( rand_path )
		
		in_data , class_list = gen_PM2_inputs( p_txts , c_txts, c_keys )

	return train_valid_test_data( np.array( in_data ) , np.array( class_list ) , len( np.array( in_data ) ) )

# Storing Results in txt files
################################################################################

def epoch_results_filewrite(results_names,lyr,wid,l_rate,hidden_act,y_out_act, init_wbs,opt, loss,low_level_results):

	"""
	epoch_results_filewrite
		Write to file the results of each epoch for a run
	"""

	with open(results_names+'/per_epoch_results.txt','a') as f:
		counter_run = 0
		for i in low_level_results:
			counter_run += 1
			counter_epc = 0
			for j in low_level_results[i]:
				counter_epc +=1
				f.write(\
				'layers|' + s(lyr)\
				+ '|width|' + s(wid) \
				+ '|rate|' + s(l_rate) \
				+ '|hidden_act|' + hidden_act \
				+ '|y_act|' + y_out_act \
				+ '|var_int|' + s(init_wbs[0]) \
				+ '|var_v0|' + s(init_wbs[1]) \
				+ '|var_v1|' + s(init_wbs[2]) \
				+ '|opt|' + s(opt) \
				+ '|loss|' + s(loss) \
				+ '|run|' + s(counter_run) \
				+ '|epc|' + s(counter_epc) \
				+ '|results|' + '|'.join(map(str,low_level_results[i][j])) \
				+ '\n' )

def reliability_results_filewrite(results_names,lyr,wid,l_rate,hidden_act,y_out_act, init_wbs, opt, loss,all_final_test_accs):
	
	"""
	reliability_results_filewrite
		Write to file the results of the final test accuracies for all x runs (reliability checks)
	"""
	
	with open(results_names+'/reliablility_results.txt','a') as f:
		f.write(\
		'layers|' + s(lyr) \
		+ '|width|' + s(wid) \
		+ '|rate|' + s(l_rate) \
		+ '|hidden_act|' + hidden_act \
		+ '|y_act|' + y_out_act \
		+ '|var_int|' + s(init_wbs[0]) \
		+ '|var_v0|' + s(init_wbs[1]) \
		+ '|var_v1|' + s(init_wbs[2]) \
		+ '|opt|' + s(opt) \
		+ '|loss|' + s(loss) \
		+ '|results|' + '|'.join(map(str,all_final_test_accs)) \
		+ '\n' )

def AUC_filewrite(results_names,lyr,wid,l_rate,hidden_act,y_out_act, init_wbs, opt, loss,AUC_results):
	with open(results_names+'/AUCscore.txt','a') as f:
		f.write(\
		'layers|' + s(lyr) \
		+ '|width|' + s(wid) \
		+ '|rate|' + s(l_rate) \
		+ '|hidden_act|' + hidden_act \
		+ '|y_act|' + y_out_act \
		+ '|var_int|' + s(init_wbs[0]) \
		+ '|var_v0|' + s(init_wbs[1]) \
		+ '|var_v1|' + s(init_wbs[2]) \
		+ '|opt|' + s(opt) \
		+ '|loss|' + s(loss) \
		+ '|results|' + '|'.join(map(str,AUC_results)) \
		+ '\n' )

# Network Functions
################################################################################

def get_btc_xy( i , btc_sz , indata , outdata ):
	
	"""
	get_btc_xy
		- Construct a data set for each batch iteration.
		- If the total dataset size is smaller than the batch size, then only one batch returned.
	"""
	
	min_b = int( i * btc_sz )
	max_b = int( ( i * btc_sz ) + btc_sz )
	x = indata[ min_b : max_b ]
	y = outdata[ min_b : max_b ]
	return x, y

def stdout_writing( btc , tot_btc, btc_acc_train, btc_acc_val, btc_cost, epc, epcs, epc_acc_train, epc_acc_val, epc_acc_test, epc_cost ):
	
	"""
	stdout_writing
		- Function to write current state of training etc. to the console where network is run.
		- TODO provide a single dict and dynamicly print the output (keys as name, values as values)
	"""
	
	# N.B. sys.stdout imported as console!
	console.write(\
	"\r%s" %\
	'B: ' + s( btc+1 ) + '/' + s(tot_btc) \
	+ ' | B Tr. acc: ' + "{:.1f}".format(btc_acc_train) \
	+ ' | B V. acc: ' + "{:.1f}".format(btc_acc_val) \
	+ " | B C: " + "{:.1f}".format( btc_cost ) \
	+ " | E: " + s( epc + 1 ) + '/' + s(epcs) \
	+ ' | E Tr. acc: ' + "{:.1f}".format( epc_acc_train ) \
	+ ' | E V. acc: ' + "{:.1f}".format( epc_acc_val ) \
	+ ' | E Ts. acc: ' + "{:.1f}".format( epc_acc_test ) \
	+ " | E C: " + "{:.1f}".format( epc_cost ) \
	)
	console.flush()

def epoch_results(run , epc , epcs , av_btc_d , e_acc , d1 , d2):
	
	"""
	epoch_results
		- Function to reduce amount of code in run_tf_nnet
		- Adds specific values to results dictionaries and exits
	"""
	
	d1['av_loss_v'] += e_acc['c_v'] / epcs
	d1['av_trn'] += e_acc['tr'] / epcs
	d1['av_val'] += e_acc['v'] / epcs
	d1['av_test'] += e_acc['ts'] / epcs

	if run not in d2.keys():
		d2[run] = {epc : [ s(e_acc['tr']) , s(e_acc['v']) , s(e_acc['ts']) , s(e_acc['c_v']) ] }
	else:
		d2[run][epc] = [ s(e_acc['tr']) , s(e_acc['v']) , s(e_acc['ts']) , s(e_acc['c_v']) ]

def optimizer_choice( opt_choice , lrn_rate ):
	
	### Define the training backprop algorithm as gradient descent using cross entropy loss
	
	if opt_choice == 'GD':
		optimiser = tf.train.GradientDescentOptimizer( learning_rate = lrn_rate )
		
	elif opt_choice == 'ADAdelta':
		optimiser = tf.train.AdadeltaOptimizer( learning_rate = lrn_rate )
		
	elif opt_choice == 'ADAgrad':
		optimiser = tf.train.AdagradOptimizer( learning_rate = lrn_rate )
		
	#elif opt_choice == 'Momentum':
	#	optimiser = tf.train.MomentumOptimizer( learning_rate = lrn_rate )
	
	elif opt_choice == 'Adam':
		optimiser = tf.train.AdamOptimizer( learning_rate = lrn_rate )
		
	elif opt_choice == 'PAD':
		optimiser = tf.train.ProximalAdagradOptimizer( learning_rate = lrn_rate )
		
	elif opt_choice == 'RMSprop':
		optimiser = tf.train.RMSPropOptimizer( learning_rate = lrn_rate )
		
	return optimiser
		
def loss_choice( choice , label , prediction ):
	
	""" 
	Choose which loss function to use
	"""
	# isn't cross entropy equation this? VVV
	# (y_expected * log(y_prediction) ) + ( 1 - y_expected ) * log( 1 - y_prediction )
	
	if choice == 'cross_entropy':
		loss = -tf.reduce_mean( tf.reduce_sum( label * tf.log( prediction ) + ( 1 - label ) * tf.log( 1 - prediction ) , axis=1 ) )

	elif choice == 'sig_ce':
		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = label,logits = prediction)

	elif choice == 'smax_ce':
		loss = tf.losses.softmax_cross_entropy(onehot_labels = label,logits = prediction)
	
	# Not using softmax for these tasks!
	#elif self.loss_choice == 'sparse_smax_ce':
	#	loss = tf.losses.sparse_softmax_cross_entropy(labels = label,logits = prediction)

	elif choice == 'mse':
		loss = tf.losses.mean_squared_error(labels = label , predictions = prediction)

	elif choice == 'hinge':
		loss = tf.losses.hinge_loss(labels = label , logits = prediction)

	elif choice == 'log':
		loss = tf.losses.log_loss(labels = label , predictions = prediction)

	elif choice == 'cosine':
		loss = tf.losses.cosine_distance(labels = label , predictions = prediction)

	return loss

def activation_choice( choice, data):

	""" 
	Choose which activation function to use
	"""

	if choice == 'sigmoid':
		return tf.nn.sigmoid(data)
	if choice == 'relu':
		return tf.nn.relu(data)
	if choice == 'tanh':
		return tf.nn.tanh(data)
	if choice == 'softmax':
		return tf.nn.softmax(data)
	if choice == 'elu':
		return tf.nn.elu(data)
	if choice == 'softsign':
		return tf.nn.softsign(data)

def init_choice(size,choice_list):
	
	""" Choose how variables (weights / biases) are initialised 
	1. Choose the type of random distribution to use
	2. Provide input numbers (mean, std.dev etc)
	"""
	
	if choice_list[0] == 'uni':
		return tf.random_uniform(shape=size, minval=choice_list[1],maxval=choice_list[2])
	elif choice_list[0] == 'norm':
		return tf.random_normal(shape=size, mean=choice_list[1],stddev=choice_list[2])
	elif choice_list[0] == 'trunc_norm':
		return tf.truncated_normal(shape=size, mean=choice_list[1],stddev=choice_list[2])
	elif choice_list[0] == 'zeroes':
		return tf.zeros(shape=size)
	elif choice_list[0] == 'constant':
		return tf.constant(shape=size,value=int(choice_list[1]),dtype=tf.float32)
	elif choice_list[0] == 'ones':
		return tf.ones(shape=size)
	else:
		print('Ahhh no! problems with init choice arguments inputs!')
		print('problem with: '+s(choice_list[0]))

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		#tf.summary.scalar('var', var)
		tf.summary.scalar('mean', mean,collections=['train','test'])
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev,collections=['train','test'])
		tf.summary.scalar('max', tf.reduce_max(var),collections=['train','test'])
		tf.summary.scalar('min', tf.reduce_min(var),collections=['train','test'])
		tf.summary.histogram('histogram', var,collections=['train','test'])

# THE NETWORK
################################################################################

def run_tf_nnet(trn_x,trn_y,val_x,val_y,test_x,test_y,result_path,param_dict = None):
	
	""" TODO UPDATE COMMENTS
	run_tf_nnet
		- Main run-time function
		- Takes A LOT of variable inputs! Can be a pain to keep track of!!!
		- Can run x times for the same parameters - to ensure reliability of results
		- Will run for n epochs, with m batches (depending on data size)
		- Most of the 'dirty' code is sotring accuracy variables. Unavoidable really.
		- When called, the function returns:
			- All x reliability runs final testing results (test dataset)
			- (Currently) the per epoch results (validation/test datasets,cost) for the LAST run.
	
	
	run_tf_nnet:
		- The big main script used to define & run neural network
		- plenty of stuff going on
		- 3 x iterables (reliability run, epochs, batches)
		- parameter dictionary used to store all the parameters for the netwrok runs
		- lots of variables defined to record the accuracies and other stats
		- also got the Tensorboard summary stuff
	
	"""


	if param_dict == None:
		
		param_dict = {\
		 'epcs' : 75 , 'btc_sz' : 1000 , 'lyr_wdth' : 10 , 'lrn_rate' : 0.5 , 'lyrs' : 1 , 'runs' : 10 \
		, 'activation_model' : 'sigmoid' , 'y_out_activation' : 'sigmoid' , 'init_choices' : ['norm',0,1] \
		, 'opt_choice' : 'GD', 'loss_choice' : 'cross_entropy' , 'weight_rw_flag' : False , 'tb_write' :  False }
		
		print("I'm guessing you're using this function standalone, rather than with the run_models main func.")
		print("You didn't give me a parameter dictionary, so I'm using a default one instead.")
		print("These are the parameters:")
		print(param_dict)

	all_final_test_accs , all_final_AUCs , all_epoch_results = list() , list() , dict(dict())
	
	for run in range(param_dict['runs']):

		""" --- MODEL
		accuracy / measurement dictionary definitions (need to reset per run)
		"""
		
		e_v_loss , e_acc, av_epc_d = 0 , {'tr':0,'v':0,'ts':0} , {'av_loss_v':0,'av_trn':0,'av_val':0,'av_test':0}
		writeflag , epoch_results_list , epoch_acc_hist = True , list() , list()
	
		""" --- MODEL
		per run variables - width of input, width of output, total batches 
		"""
		weight_initialisations = [param_dict['init_choices'][0] , param_dict['init_choices'][1],param_dict['init_choices'][2] ]
		if len(param_dict['init_choices']) == 4:
			bias_initialisations = [param_dict['init_choices'][0] , param_dict['init_choices'][3],param_dict['init_choices'][3] ]
		else:
			bias_initialisations = [param_dict['init_choices'][0] , param_dict['init_choices'][1],param_dict['init_choices'][2] ]
		x_width, y_width , tot_btc = len(trn_x[0]) , len(trn_y[0]) , int(len(trn_x)/param_dict['btc_sz'])

		""" --- MODEL
		define a tensorflow session - allow growth to reduce memory usage
		"""
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		sess = tf.Session(config=config)

		""" --- MODEL
		Define placeholders as width of x or y wide, infite depth
		Define dictionarys to allow for dynamic network generation
		Dynamic generation was gracefully received from John McGouran
		Works by iteratively creating tf variables and storing in a dictionary
		"""
		
		x , y = tf.placeholder(tf.float32, [None,x_width]) , tf.placeholder(tf.float32, [None,y_width])
	
		W, b ,lyrs = {} , {} ,{}
		
		""" --- INPUT LAYER
		1. Assign whole layer to Tensorboard Summary
			2a. Assign weights to Tb summary
			2b. First layer width = x inputs wide -> first hidden layer wide
			2c. Initialise values with weight_initialisations variable (type of distribution, value 1, value 2)
			2d. Define which stats to collect for Tb
		3. Repeat 2 + 3 for biases (first hidden layer wide, bias_initialisations) and layer 
		Where layer = activation_fx( (( x dot input layer weights ) + input layer biases) )
		"""
		
		with tf.name_scope('layer0'):
		
			with tf.name_scope('weights'):
				W[0] = tf.Variable(init_choice([x_width, param_dict['lyr_width']], weight_initialisations), name='W0' )
				variable_summaries(W[0])
			
			with tf.name_scope('bias'):
				b[0]= tf.Variable(init_choice([param_dict['lyr_width']], bias_initialisations), name='b0' )	
				variable_summaries(b[0])
			
			with tf.name_scope('w_x_b'):
				lyrs[0] = activation_choice( param_dict['activation_model'] , tf.add( tf.matmul( x , W[0] ) , b[0] ) )
				variable_summaries(lyrs[0])
			
		""" --- Hidden layers
		As per input layer, but is current layer -> next hidden layer
		"""
		
		for h_layer_idx in range(1,param_dict['lyrs']):
		
			with tf.name_scope('layer' + s(i)):
			
				with tf.name_scope('weights'):
					W[h_layer_idx] = tf.Variable(\
					init_choice([param_dict['lyr_width'], param_dict['lyr_width']], weight_initialisations), name='W' + s(h_layer_idx) \
					)
					variable_summaries(W[h_layer_idx])
				
				with tf.name_scope('bias'):
					b[h_layer_idx] = tf.Variable(init_choice([param_dict['lyr_width']], bias_initialisations), name='b' + s(h_layer_idx) )
					variable_summaries(b[h_layer_idx])
				
				with tf.name_scope('w_x_b'):
					lyrs[h_layer_idx] = activation_choice( \
					param_dict['activation_model'] , tf.add( tf.matmul( lyrs[h_layer_idx-1], W[h_layer_idx] ) , b[h_layer_idx]) )
					variable_summaries(lyrs[h_layer_idx])
		
		""" --- Output layer 
		As per input layer, but is last hidden layer -> output
		"""
		
		with tf.name_scope('layer' + s(param_dict['lyrs'])):
		
			with tf.name_scope('weights'):
				W[param_dict['lyrs']] = tf.Variable(\
				init_choice([param_dict['lyr_width'], y_width], weight_initialisations), name='W' + s( param_dict['lyrs'] ) \
				)
				variable_summaries(W[param_dict['lyrs']])
			
			with tf.name_scope('bias'):
				b[param_dict['lyrs']] = tf.Variable(init_choice([y_width], bias_initialisations), name='b' + s( param_dict['lyrs'] ) )
				variable_summaries(b[param_dict['lyrs']])
			
			with tf.name_scope('w_x_b'):		
				lyrs[param_dict['lyrs']] \
				= activation_choice( param_dict['y_out_activation']\
				, tf.add( tf.matmul( lyrs[ param_dict['lyrs']-1 ], W[ param_dict['lyrs'] ] ), b[ param_dict['lyrs'] ] )\
				)
				variable_summaries(lyrs[param_dict['lyrs']])
		
		""" --- PREDICTIONS
		Clip the y output value
		If y < 1e-10, set = 0, if y > 0.999999, set to 1
		this is left over from the original codebase
		N.B. THIS GOT CHANGED RECENTLY AND NEEDS TESTING
		"""
		
		with tf.name_scope('predictions'):
			y_clipped = tf.clip_by_value(lyrs[param_dict['lyrs']], 0.49999999, 0.5)
			variable_summaries(y_clipped)
		
		""" --- CORRECT PREDICTIONS
		What is a correct prediction ?
		It's when the rounded Y_output is equal to the labels (y)
		correct_prediction is 1 when true, 0 when false
		Rounding may be depreciated depending on change to y_clipped clipping values
		"""
		
		with tf.name_scope('correct_predictions'):
			correct_prediction = tf.equal( y , tf.round( y_clipped ))
		
		""" Front top of Tb output, important things to measure"""
		
		with tf.name_scope('general'):
			
			""" --- ACCURACY
			Accuracy = Correct predictions / number of total predictions
			1. Cast the correct_prediction into floats
			2. Calculate the mean (i.e. Number of TRUE values / number of values )
			3. Multiply by 100 to get useful accuracy
			"""
			
			with tf.name_scope('accuracy'):
				accuracy = tf.multiply( tf.reduce_mean( tf.cast( correct_prediction , tf.float32 ) ) , 100 )
				tf.summary.scalar('accuracy', accuracy,collections=['train','test'])
		
			""" --- LOSS FUNCTION
			Choose loss function using loss_choice func. + value from parameter dict.
			Give it the labels + predictions so it can work out error
			"""
			
			with tf.name_scope('loss'):
				loss = loss_choice( param_dict['loss_choice'], y , y_clipped )
				tf.summary.scalar('loss',loss,collections=['train'])

			""" -- OPTIMISER
			How the network is trained
			Choose an optimiser using optimizer_choice func. + value from parameter dict.
			Pass the learning rate to optimiser
			Tell it we want to minimise loss function (error) 
			"""
		
			with tf.name_scope('optimiser'):
				optimiser = optimizer_choice( param_dict['opt_choice'] , param_dict['lrn_rate'] ).minimize( loss )


		""""		
		Collect all the Tesorboard summaries together
		We have two here, so we can measure things for training and testing seperately
		If this is a basic run:
			Create directories for the Tb summary files (if they don't already exist)
			Define where to write the files to
			n.b. training files keep the graph's design information (sess.graph assignment)
		"""
				
		merged_train = tf.summary.merge_all('train')
		merged_test = tf.summary.merge_all('test')
	
		if param_dict['tb_write'] is True:
	
			#run_path =  s(param_dict['lyrs']) + 'lyr_' + s(param_dict['lyr_width']) + 'wid_' +  s(param_dict['lrn_rate']) + 'lrate_' \
			#+ param_dict['activation_model'] + 'hact_' + param_dict['y_out_activation'] + 'yact_' \
			#+ param_dict['opt_choice'] + 'opt_' + s(param_dict['loss_choice']) + 'loss_' + 'run' + s(run)
	
			if not os.path.exists(result_path+ '/batchtrain' ): #+ run_path):
				os.makedirs(result_path + '/batchtrain') #+ run_path)
				os.makedirs(result_path + '/batchvalid') #+ run_path)
				os.makedirs(result_path + '/epochvalid') #+ run_path)
				os.makedirs(result_path + '/epochtest') #+ run_path)
			
			btc_trn_writer = tf.summary.FileWriter(result_path + '/batchtrain',sess.graph) #+ run_path ,sess.graph)
			btc_val_writer = tf.summary.FileWriter(result_path + '/batchvalid') #+ run_path)
			epc_val_writer = tf.summary.FileWriter(result_path + '/epochvalid') #+ run_path)
			epc_test_writer = tf.summary.FileWriter(result_path + '/epochtest') #+ run_path)
	
	
		""" ---- MODEL
		initialise all the weigfhts within the session using tf.g_v_i() initisliser
		"""
			
		sess.run( tf.global_variables_initializer() )

		""" --- MODEL
		print info about net being run 
		"""
		
		print( '\n' \
		+ 'Lyr width: ' + s( param_dict['lyr_width'] ) \
		+ ' | Lyrs: ' + s( param_dict['lyrs'] ) \
		+ ' | Lrn Rate: ' + s(param_dict['lrn_rate']) \
		+ ' | Run: ' + s(run) \
		+ ' | hidden_act: ' + param_dict['activation_model'] \
		+ ' | y_act: ' + param_dict['y_out_activation'] )
	
		print( ' | results path: ' + result_path \
		+ ' | inits: ' + '|'.join(map(str,param_dict['init_choices'])) \
		+ ' | opt. ch.: ' + param_dict['opt_choice'] \
		+ ' | loss ch.: ' + param_dict['loss_choice'] \
		)

		""" --- RUN
		TRAINING VALIDATION AND TESTING
		"""
		
		for epc_idx in range(param_dict['epcs']):

			""" ---EPOCH
			define average batch dictionary (needs to reset on new epoch) 
			"""
			
			av_btc_d = { 'av_c' : 0 , 'av_trn' : 0 , 'av_val' : 0 }
			
			for batch_idx in range(tot_btc):

				"""  --- BATCH
				Get batch data 
				"""
				
				btc_x , btc_y = get_btc_xy(batch_idx , param_dict['btc_sz'] , trn_x , trn_y)
				
				#"""
				# TF can output metadata about the network is running
				# This didn't get completed
				#"""
				
				#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				#run_metadata = tf.RunMetadata()
			
				"""  --- BATCH
				TRAIN model, get stats out, test with validation set
				"""
				
				summary1 , _ , c, b_tr = sess.run([merged_train , optimiser , loss , accuracy] ,feed_dict = {x : btc_x , y : btc_y} )
				# See above comment on metadata
				#, options=run_options,  run_metadata=run_metadata )
				summary2, b_v = sess.run([merged_test, accuracy] , feed_dict = {x : val_x , y : val_y} )
				
				""" --- BATCH
				store results results 
				"""
				
				btc_acc = {'tr' : b_tr , 'v' : b_v }
				av_btc_d['av_c'] += (c / tot_btc)
				av_btc_d['av_trn'] += btc_acc['tr'] / tot_btc
				av_btc_d['av_val'] += btc_acc['v'] / tot_btc
				
				""" --- TB output
				Write Tb summaries to the batch files
				"""
				
				if param_dict['tb_write'] is True:
					btc_trn_writer.add_summary(summary1,epc_idx*(tot_btc-1)+i)
					btc_val_writer.add_summary(summary2,epc_idx*(tot_btc-1)+i)
				
				""" --- BATCH
				write ALL results to console (incl. prev. epoch's results) 
				"""
				
				stdout_writing(\
				batch_idx,tot_btc,btc_acc['tr'],btc_acc['v'],c,epc_idx,param_dict['epcs'],e_acc['tr'],e_acc['v'],e_acc['ts'],e_v_loss\
				)
				
			""" --- EPOCHS
			Run EPOCH validation and testing 
			"""
				
			summary3, e_v, e_v_loss = sess.run([merged_test, accuracy , loss ] , feed_dict = {x : val_x , y : val_y} )
			summary4, e_ts = sess.run( [merged_test, accuracy] , feed_dict = {x : test_x , y : test_y} )

			""" --- EPOCHS
			Store EPOCH results 
			"""
			
			e_acc = {'tr' : av_btc_d['av_trn'] , 'v' : float(e_v) , 'ts' : float(e_ts) , 'c_v' : float(e_v_loss) }
			epoch_results(run , epc_idx , param_dict['epcs'] , av_btc_d , e_acc , av_epc_d , all_epoch_results)
			epoch_acc_hist.append(e_ts)

			""" --- TB output
			Write Tb summaries to the batch files
			"""
		
			if param_dict['tb_write'] is True:
				epc_val_writer.add_summary(summary3,epc_idx)
				epc_test_writer.add_summary(summary4,epc_idx)

			""" Checks on network behaviour """

			if epc_idx > 10:
				
				if (b_tr - 15) > e_v or (c - 40) > e_v_loss:

					""" --- Overfitting check
					check if training acc/loss has run away from validation acc/loss 
					"""
					
					print('\nEnding run, model is overfitting!\n')
					writeflag = False
					break
				
				
				elif np.mean(epoch_acc_hist[epc-10:epc]) > 90:
					
					""" --- Got a good result?
					check if average of last 10 epochs is over 90% 
					If so, quit - we've got a good result
					don't need to waste time on more epochs
					"""
					
					print('\nEarly finish, average of last 10 epochs over 90%\n')
					break

		""" --- RUNS
		Final run clean up / results write out (AUC etc)
		"""
		
		if param_dict['tb_write'] is True:
			btc_trn_writer.close()
			btc_val_writer.close()
			epc_val_writer.close()
			epc_test_writer.close()
		
		
		""" --- OVERFITTING ESCAPE
		write flag decides if we're writing results to disk or not
		If overfitting, then write the parameters to overfitting file
		Else, store the results
		"""
		
		if writeflag is True:
			
			""" --- RESULTS STORAGE
			Per run final accuracy testing (and storage of result)
			Per run AUC results - N.B. a bug in tf requires sess.run(local variables init)
			Only run AUC for binary classification problems
			"""
			
			# ------ Save the model's variables
			#save_path = saver.save(sess, result_path + "_model" + s(run) + ".ckpt")
			#print("\nModel saved in file: %s" % save_path + '\n')

			# ------ Write the final labels + predictions combo to disk (delete cos could get big!)
			#in_outs = sess.run(model.outs,feed_dict = {x : test_x , y : test_y} )
			#tf.write_file(filename = result_path + "_model" + s(run) + '_' + 'in_out.txt' , contents = in_outs)
			##tf.write_file(filename = result_path + "_model" + s(run) + '_' + 'predictions.txt' , contents = predictions)
			#del in_outs
		
			if y_width == 1:
				auc_fx = tf.metrics.auc(y, y_clipped)
				sess.run( tf.local_variables_initializer() )
				acc , auc = sess.run([accuracy , auc_fx],feed_dict = {x : test_x , y : test_y})
				all_final_test_accs.append(acc)
				all_final_AUCs.append(auc[1])
			else:
				acc = sess.run(accuracy,feed_dict = {x : test_x , y : test_y} )
				all_final_test_accs.append(acc)
				all_final_AUCs.append([y,y_clipped])

		elif writeflag is False:
			
			""" --- OVERFITTING
			Write info about the net to disk
			Put data in the results variables
			0 is checked for in the run_models.py, so careful about changing that
			"""
			
			with open(result_path + '_overfits.txt','a') as f:
				f.write( \
				'Lyr width|' + s( param_dict['lyr_width'] ) \
				+ '|Lyrs|' + s( param_dict['lyrs'] ) \
				+ '|Lrn Rate|' + s(param_dict['lrn_rate']) \
				+ '|Run|' + s(run) \
				+ '|hidden_act|' + param_dict['activation_model'] \
				+ '|y_act: ' + param_dict['y_out_activation'] \
				+ '|inits|' + '|'.join(map(str,param_dict['init_choices'])) \
				+ '|opt. ch.|' + param_dict['opt_choice'] \
				+ '|loss|' + param_dict['loss_choice'] \
				+ '|e_v|' + s(e_v) \
				+ '|btc_tr|' + s(b_tr) \
				+ '|av_btc_tr|' + s(av_btc_d['av_trn']) \
				+ '|e_loss|' + s(e_v_loss) \
				+ '|btc_loss|' + s(c) \
				+ '\n' )
			all_final_test_accs.append(0)
			all_final_AUCs.append(0)
		
		"""
		Reset the graph
		This has to be here when using Tesnorboard summaries
		For some reason, it doesn't reset the graph otherwise
		(probably beacause they expect people to save the graph using checkpoints)
		If Tensorboard code is removed, this isn't needed
		"""	
		
		tf.reset_default_graph()
	
	""" Return results data to whichever function called me """
	
	return all_final_test_accs, all_epoch_results, all_final_AUCs

# Prepare runs in main
################################################################################

def main(args):

	"""
	Main 'get data and run networks' function
	"""

	working_directory_check()

	results_names = 'Code/neural-nets/tensorboard_design/results/'+args.results_string
	if not os.path.exists(results_names + '/'):
		os.makedirs(results_names + '/')

	with open(results_names+'/argsofrun.txt','w') as f:
		f.write(s(args) + '\n')

	if args.run_type == 'basic':

		inputs_train, outputs_train, inputs_valid, outputs_valid, inputs_test, outputs_test = \
		get_and_prepare_data( \
		args.problem \
		, args.p_txt_path \
		, args.c_txt_dir \
		, args.rand_path \
		, data_size = args.datasize \
		)

		init_values =  [ args.inits , args.init_vars[0] , args.init_vars[1] ]
		
		p_dict = { \
			'lyr_width' : args.width \
			, 'epcs' : args.epochs \
			, 'lrn_rate' : args.lrate \
			, 'lyrs' : args.layers \
			, 'btc_sz' : args.batch_size \
			, 'runs' : args.runs \
			, 'activation_model' : args.hacts \
			, 'y_out_activation' : args.yacts \
			, 'init_choices' : init_values \
			, 'opt_choice' : args.optimiser \
			, 'loss_choice' : args.loss \
			,  'weight_rw_flag' : True \
			,  'tb_write' : True \
			}
		
		all_final_test_accs, low_level_results, AUC_results = run_tf_nnet( \
		inputs_train \
		, outputs_train \
		, inputs_valid \
		, outputs_valid \
		, inputs_test \
		, outputs_test \
		, results_names \
		, param_dict=p_dict \
		)
	
		# results write out
		
		reliability_results_filewrite( \
		results_names \
		, args.layers \
		, args.width \
		, args.lrate \
		, args.hacts \
		, args.yacts \
		, init_values \
		, args.optimiser \
		, args.loss \
		, all_final_test_accs \
		)

		epoch_results_filewrite( \
		results_names \
		, args.layers \
		, args.width \
		, args.lrate \
		, args.hacts \
		, args.yacts \
		, init_values \
		, args.optimiser \
		, args.loss \
		, low_level_results \
		)
		if args.problem not in ['P1','P2']:
			AUC_filewrite( \
			results_names \
			, args.layers \
			, args.width \
			, args.lrate \
			, args.hacts \
			, args.yacts \
			, init_values \
			, args.optimiser \
			, args.loss \
			, AUC_results \
			)

	elif args.run_type == 'grid_search':

		inputs_train, outputs_train, inputs_valid, outputs_valid, inputs_test, outputs_test = \
		get_and_prepare_data( \
		args.problem \
		, args.p_txt_path \
		, args.c_txt_dir \
		, args.rand_path \
		, data_size = args.datasize \
		)

		lrate_choices = [ float(i)/10 for i in range( int( min(args.lrate) * 10 ) , int(  max(args.lrate) * 10) + 1  )]
		width_choices = [ j for j in range( min(args.width) , max(args.width) + 1 ) ]
		act_layer_choices = [ k for k in range(min(args.layers),max(args.layers)+1) ]

		for y_out_act in args.yacts:
			for hidden_act in args.hacts:
				for lyr in act_layer_choices:
					for wid in width_choices:
						for l_rate in lrate_choices:
							for opt in args.optimiser:
								for loss in args.loss:
									for init in args.inits:
										
										init_wbs = [ init , args.init_vars[0] , args.init_vars[1] ]
										
										p_dict = { \
											'lyr_width' : wid \
											, 'epcs' : args.epochs \
											, 'lrn_rate' : l_rate \
											, 'lyrs' : lyr \
											, 'btc_sz' : args.batch_size \
											, 'runs' : args.runs \
											, 'activation_model' : hidden_act \
											, 'y_out_activation' : y_out_act \
											, 'init_choices' : init_wbs \
											, 'opt_choice' : opt \
											, 'loss_choice' : loss \
											,  'weight_rw_flag' : False \
											,  'tb_write' : False \
											}
										
										all_final_test_accs, low_level_results, AUC_results = run_tf_nnet( \
										inputs_train \
										, outputs_train \
										, inputs_valid \
										, outputs_valid \
										, inputs_test \
										, outputs_test \
										, results_names \
										, param_dict = p_dict \
										)

										# results write out
										reliability_results_filewrite( \
										results_names \
										, lyr \
										, wid \
										, l_rate \
										, hidden_act \
										, y_out_act \
										, init_wbs \
										, opt \
										, loss \
										, all_final_test_accs \
										)

										epoch_results_filewrite( \
										results_names \
										, lyr \
										, wid \
										, l_rate \
										, hidden_act \
										, y_out_act \
										, init_wbs \
										, opt \
										, loss \
										, low_level_results \
										)
										
										if args.problem not in ['P1','P2']:
											AUC_filewrite( \
											results_names \
											, lyr \
											, wid \
											, l_rate \
											, hidden_act \
											, y_out_act \
											, init_wbs \
											, opt \
											, loss \
											, AUC_results \
											)

	elif args.run_type == 'random_search':

		inputs_train, outputs_train, inputs_valid, outputs_valid, inputs_test, outputs_test = \
		get_and_prepare_data( \
		args.problem \
		, args.p_txt_path \
		, args.c_txt_dir \
		, args.rand_path \
		, data_size = args.datasize \
		)
		
		
		act_layer_choices = [ k for k in range(min(args.layers),max(args.layers)+1) ]

		for its in range(args.popt_iter):

			wid = randint( min(args.width) , max(args.width) )
			l_rate = randu( min(args.lrate) , max(args.lrate) )
			hidden_act = args.hacts[ randint( 0 , len( args.hacts ) -1 ) ]
			y_out_act = args.yacts[ randint( 0 , len( args.yacts ) -1 ) ]

			opt = args.optimiser[ randint( 0 , len( args.optimiser ) -1 ) ]
			lss = args.loss[ randint( 0 , len( args.loss ) -1 ) ]
			lyr = act_layer_choices[ randint( 0 , len( act_layer_choices ) -1 ) ]

			if len(args.init_vars) == 2:
				init_wbs = [ \
						args.inits[ randint( 0 , len( args.inits ) -1 ) ] \
						, randu( args.init_vars[0] , args.init_vars[1] ) \
						, randu( args.init_vars[0] , args.init_vars[1] ) \
						]
			elif len(args.init_vars) == 4:
				init_wbs = [ \
						args.inits[ randint( 0 , len( args.inits ) -1 ) ] \
						, randu( args.init_vars[0] , args.init_vars[1] ) \
						, randu( args.init_vars[2] , args.init_vars[3] ) \
						]

			p_dict = { \
				'lyr_width' : wid \
				, 'epcs' : args.epochs \
				, 'lrn_rate' : l_rate \
				, 'lyrs' : lyr \
				, 'btc_sz' : args.batch_size \
				, 'runs' : args.runs \
				, 'activation_model' : hidden_act \
				, 'y_out_activation' : y_out_act \
				, 'init_choices' : init_wbs \
				, 'opt_choice' : opt \
				, 'loss_choice' : lss \
				,  'weight_rw_flag' : False \
				,  'tb_write' : False \
				}

			all_final_test_accs, low_level_results, AUC_results = run_tf_nnet( \
			inputs_train , outputs_train \
			, inputs_valid , outputs_valid \
			, inputs_test , outputs_test \
			, results_names , param_dict = p_dict )

			# results write out
			reliability_results_filewrite( \
			results_names \
			, lyr \
			, wid \
			, l_rate \
			, hidden_act \
			, y_out_act \
			, init_wbs \
			, opt \
			, lss \
			, all_final_test_accs \
			)

			epoch_results_filewrite( \
			results_names \
			, lyr \
			, wid \
			, l_rate \
			, hidden_act \
			, y_out_act \
			, init_wbs \
			, opt \
			, lss \
			, low_level_results \
			)
			
			if args.problem not in ['P1','P2']:
				AUC_filewrite( \
				results_names \
				, lyr \
				, wid \
				, l_rate \
				, hidden_act \
				, y_out_act \
				, init_wbs \
				, opt \
				, lss \
				, AUC_results \
				)


	elif args.run_type == 'datasizes':

		data_sizes = [i*100000 for i in range(1,int(args.datasize/100000))]
		for data_size in data_sizes:

			inputs_train, outputs_train, inputs_valid, outputs_valid, inputs_test, outputs_test = \
			get_and_prepare_data( \
			args.problem \
			, args.p_txt_path \
			, args.c_txt_dir \
			, args.rand_path \
			, data_size = data_size \
			)


			p_dict = { \
				'lyr_width' : args.width \
				, 'epcs' : args.epochs \
				, 'lrn_rate' : args.lrate \
				, 'lyrs' : args.layers \
				, 'btc_sz' : args.batch_size \
				, 'runs' : args.runs \
				, 'activation_model' : args.acts \
				, 'y_out_activation' : 'sigmoid' \
				, 'init_choices' : [ args.inits , args.init_vars[0], args.init_vars[1] ] \
				}

			########################################################################################

			all_final_test_accs, low_level_results, AUC_results = run_tf_nnet( \
			inputs_train \
			, outputs_train \
			, inputs_valid \
			, outputs_valid \
			, inputs_test \
			, outputs_test \
			, results_names \
			, p_dict \
			)

			# data size checks
			with open(results_names+'_per_epoch_datasizes_tests.txt','a') as f:
				counter_run = 0
				for i in low_level_results:
					counter_run += 1
					counter_epc = 0
					for j in low_level_results[i]:
						counter_epc +=1
						f.write('datasize|'+s(data_size) + '|run|' + s(counter_run) + '|epc|' + s(counter_epc) + '|results|' + '|'.join(low_level_results[i][j]) + '\n')

	elif args.run_type == 'timetaken':

		inputs_train, outputs_train, inputs_valid, outputs_valid, inputs_test, outputs_test = \
		get_and_prepare_data( \
		args.problem \
		, args.p_txt_path \
		, args.c_txt_dir \
		, args.rand_path \
		, data_size = args.datasize \
		)

		for i in range(args.popt_iter):

			starttime = dt.now()

			data_size = randint(1,args.datasize)
			batch_size = randint(1,args.batch_size)

			########################################################################################

			p_dict = { \
				'lyr_width' : args.width \
				, 'epcs' : args.epochs \
				, 'lrn_rate' : args.lrate \
				, 'lyrs' : args.layers \
				, 'btc_sz' : batch_size \
				, 'runs' : args.runs \
				, 'activation_model' : args.acts \
				, 'y_out_activation' : 'sigmoid' \
				, 'init_choices' : [ args.inits , args.init_vars[0], args.init_vars[1] ] \
				}


			all_final_test_accs, low_level_results, AUC_results = run_tf_nnet( \
			inputs_train[0:data_size] \
			, outputs_train[0:data_size] \
			, inputs_valid[0:data_size] \
			, outputs_valid[0:data_size] \
			, inputs_test[0:data_size] \
			, outputs_test[0:data_size] \
			, results_names \
			, param_dict = p_dict \
			)

			timetaken = dt.now() - starttime

			result = 'datasize|' + s(data_size) \
			+ '|batches|' + s(batch_size) \
			+ '|timetaken|' + s(timetaken)\
			+ '|result(s)|' + '|'.join(map(str,all_final_test_accs)) \
			+ '\n'
			print('\n'+result)
			# data size checks
			with open(results_names+'_time_taken.txt','a') as f:
				f.write(result)

# Argparse functionality
################################################################################

if __name__ == '__main__':

	import argparse

	"""
	Gather args passed to script from command line and run main func
	"""

	parser = argparse.ArgumentParser()

	dir_args = parser.add_argument_group('Directories','Arguments for passing file/directory paths')
	runtime_args = parser.add_argument_group('Run time properties','Arguments for how to run the nnet')
	network_args = parser.add_argument_group('Network properties','Arguments for how the nnet is built')

	network_args.add_argument('--layers', help="max layers (for param optimiseenter two numbers)", nargs='+', dest='layers', default=1 ,type=int)
	network_args.add_argument('--width' ,help="widths of layers. for popt, provide 2 entries: start end",nargs='+',dest='width',default=5,type=int)
	network_args.add_argument('--lrate',help="learning rate. for popt, provide 2 entries: start end",nargs='+',dest='lrate',default=0.5,type=float)
	network_args.add_argument('--hacts', help="choose h act. funcs. of net. (default sigmoid)", choices = ['sigmoid','tanh','relu','softmax','softsign','elu'] , nargs='+',dest='hacts',default='signoid',type=str)
	network_args.add_argument('--yacts', help="choose y act. funcs. of net. (default sigmoid)", choices = ['sigmoid','tanh','relu','softmax','softsign','elu'] , nargs='+',dest='yacts',default='signoid',type=str)
	network_args.add_argument('-o','--optimiser',help="choose optimiser", choices = ['Adam','ADAgrad','ADAdelta','GD','PAD','RMSprop'],nargs='+', dest = 'optimiser', default='GD',type=str)
	network_args.add_argument('--loss',help="choose loss func.", choices = [ 'cosine','log','hinge','mse','smax_ce','sig_ce','cross_entropy' ],nargs='+', dest = 'loss',default='cross_entropy',type=str)
	network_args.add_argument('-i','--inits',help="type of weights init", choices = ['norm','uni','trunc_norm','zeroes','constant','ones'],nargs='+', dest='inits',default='norm',type=str)
	network_args.add_argument('--init_vars', help="Weights, bias inital values", nargs='+', dest='init_vars',default=[0,1],type=float)

	dir_args.add_argument('p_txt_path', help="<R> Plain texts bin DIRECT filepath",type=str)
	dir_args.add_argument('rand_path', help="<R> Eithe the random texts bin OR key bin DIRECT filepath",type=str)
	dir_args.add_argument('c_txt_dir', help="<R> Cipher text directory",type=str)
	dir_args.add_argument('results_string', help="<R> Results folder name",type=str)

	runtime_args.add_argument('problem', help="<R> Which task (C1,C2,P1,P2)", choices = ['C1','C2','P1','P2'],type=str)
	runtime_args.add_argument('-e','--epochs', help="Choose max. number of epochs (default 100)",dest='epochs',default=100,type=int)
	runtime_args.add_argument('-b','--batch_size', help="Choose batch sizes (default 20k)",dest='batch_size',default=20000,type=int)
	runtime_args.add_argument('-d','--datasize', help="No. of data points to use (0->x), uses all if not provided",default = 0,type=int)
	runtime_args.add_argument('-r','--run_type', help="Type of run (basic, grid_search, random_search, datasizes, timetaken) (default basic)", choices = ['basic','grid_search','random_search','datasizes','timetaken'],dest='run_type',default='basic' ,type=str)
	runtime_args.add_argument('--runs', help="Number of reliability runs (default 1)",dest='runs',default=1,type=int)
	runtime_args.add_argument('--popt_iter', help="how many popt random iterations to do",dest='popt_iter',default=10,type=int)

	args = parser.parse_args()

	# ------------ check for problems

	checklist = [str,float,int]

	if len(args.init_vars) < 2:
		print(args.init_vars)
		print('Wrong number of initialisation variables provided, give 2 (basic runs) or 4 (popt runs)')
		exit()

	if args.run_type in ['grid_search','random_search']:
		if type(args.layers) in checklist:
			args.layers = [ args.layers ]
		if type(args.width) in checklist:
			args.width = [ args.width ]
		if type(args.lrate) in checklist:
			args.lrate = [ args.lrate ]
		if type(args.hacts) in checklist:
			args.hacts = [ args.hacts ]
		if type(args.lrate) in checklist:
			args.yacts = [ args.yacts ]
		if type(args.optimiser) in checklist:
			args.optimiser = [ args.optimiser ]
		if type(args.loss) in checklist:
			args.loss = [ args.loss ]
		if type(args.inits) in checklist:
			args.inits = [ args.inits]
		if len(args.init_vars) == 3:
			args.init_vars = args.init_vars[0:1]
			print('Wrong number of initialisation variables provided, using firast two')
		elif len(args.init_vars) > 4:
			args.init_vars = args.init_vars[0:3]
			print('Wrong number of initialisation variables provided, using firast four')

		for i in args.layers:
			if i < 1:
				print("can't have a network with less than 1 width")
				exit()
		for i in args.width:
			if i < 1:
				print("can't have a network with less than 1 width")
				exit()
		for i in args.lrate:
			if i > 1 or i < 0:
				print("incorrect param optimisation learning rate argument")
				exit()
	else:
		if type(args.layers) not in checklist:
			args.layers = args.layers[0]
		if type(args.width) not in checklist:
			args.width = args.width[0]
			if args.width < 1:
				print("can't have a network with less than 1 layer!")
				exit()
		if type(args.lrate) not in checklist:
			args.lrate = args.lrate[0]
			if args.lrate > 1 or args.lrate < 0:
				exit()
		if type(args.hacts) not in checklist:
			args.hacts = args.hacts[0]
		if type(args.yacts) not in checklist:
			args.yacts = args.yacts[0]
		if type(args.optimiser) not in checklist:
			args.optimiser = args.optimiser[0]
		if type(args.loss) not in checklist:
			args.loss = args.loss[0]
		if type(args.inits) not in checklist:
			args.inits = args.inits[0]
		if len(args.init_vars) not in [2,4]:
			print('Quitting, wrong number of initialisation variables provided. Need 2 or 4 (weights, biases).')
			exit()

	print('\n---------++++++++++++---------\n')
	print(args)
	print('\n---------++++++++++++---------\n')

	del checklist
	main(args)
