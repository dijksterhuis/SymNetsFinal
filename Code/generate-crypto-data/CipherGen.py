from itertools import product
from random import randint, sample
from joblib import Parallel, delayed
from sys import stdout as console
import argparse, os

################################################################################################################################################
# Functions used by the 3 main cipher functions
################################################################################################################################################

def s(string):
	"""
	# Used by all
	# Return a string
		# Used for cleaner + smaller code
	"""
	return str(string)

def mtp_fullent_file_write(filepath,data):
	"""
	# Used by MTP only
	# Write data from a given input list to txt file
	"""
	with open(filepath,'w') as f:
		for i in data:
			f.write(''.join(map(str,i)) +'\n')

def randomise(data):
	"""
	# Used by mtp only
	# Do random sampling of data and return a subset the same size as the length of all the data
		# i.e. Randomise the input list!
	"""
	return sample(data,len(data))

def XOR_bits(plain_entry,pad_entry,blk_sz):
	
	"""
	# used by mtp only
	# Do the XORing:
		# If plan text bit == pad bit then set to 0
		# Else set to 1
	"""
	return [ 0 if plain_entry[bit_idx] == pad_entry[bit_idx] else 1 for bit_idx in range(blk_sz) ]

def plain_rands(data,max_data_sz=1000000):
	
	"""
	# Used by mtp only
	# Function to do random sampling on a list...
		# By doing random sampling on a list for the list length, we get a randomised list!
	"""
	
	if max_data_sz == None:
		x, y = randomise(data), randomise(data)
	else:
		x, y = randomise(data)[0:max_data_sz], randomise(data)[0:max_data_sz]
	return x, y

def give_list(listy):
	"""
	# used by mtp only - for joblib processing
	# Function to create a list
		# Required for the joblib parallel processing
		# Need to pass an actual function rather than a method (?)
	"""
	return list(listy)
	
def check_folders(outputdir,cipher,blk_size):
	"""
	# used by ALL
	# Check if required directories exit
		# If not, create them
	"""
	
	outputdir_string = outputdir.rstrip('/') + '/' + cipher + s(blk_size) + '/'
	
	if not os.path.exists(outputdir_string):
		os.makedirs(outputdir_string + 'plain/')
		os.makedirs(outputdir_string + 'cipher/')
		os.makedirs(outputdir_string + 'keys/')
		
	return outputdir_string

def to_byte_fromint_subs(string):
	"""
	# Used by subs
	# Convert a string representation of binary (byte array) into actual string
		# '0b00001' to '000001'
	"""
	split_byte = string.split('0b')
	padded_byte = ''.join( [''.join(['0' for i in range(0,8 - len(j))]) + j for j in split_byte[1:len(split_byte)]])
	return padded_byte

def to_byte_fromint_caesar(string,blk_sz):
	"""
	# Used by caesar
	# Convert a string representation of binary (byte array) into actual string
		# '0b00001' to '000001'
	"""
	split_byte = string.split('0b')
	byte_pad = ''.join(['0' for i in range(0,blk_sz - len(split_byte[1]))])
	padded_byte = byte_pad + split_byte[1]
	return padded_byte

def to_byte(string):
	"""
	# used by Subs only
	# convert ascii strings into bytearrays
	"""
	byte_vals = ''.join(map(bin,bytearray(string,'utf8'))).split(' ')
	return byte_vals[0]

def file_write(location,data):
	"""
	# Used by subs and caesar
	# write a line of data to file
	"""
	with open(location,'a') as f:
		f.write(data + '\n')
		
def rand_string_gen(blk_sz,symbols):
	"""
	# Used by subs only
	# choose a random set of symbols to use
	"""
	rand_ints = [ randint(0,len(symbols)-1) for idx in range(0,blk_sz) ]
	return ''.join([symbols[idx] for idx in rand_ints])
	
	
def caesar_plain_n_rand_gen(datasize,max_val,min_val,blk_sz):

	"""
	# used by caesar only
	# generate x random integers and return in a list as binary strings
	"""

	counter, full_bits_list = 0, list()

	while counter < datasize:
		full_bits_list.append( to_byte_fromint_caesar( bin( randint( min_val, max_val )), blk_sz) )
		counter +=1
		perc_compl = (counter * 100 / datasize)
		console.write("\r %s" % 'complete: '+ s(perc_compl)[0:5] +'%' + ' | - count: '+ str(counter))
		console.flush()
		
	return full_bits_list

################################################################################################################################################
# MTP script
################################################################################################################################################

def mtp(args):
	
	"""
	Joblib's Parallel provides multi-processor support
	"""
	
	with Parallel(n_jobs=4,verbose = 2) as parallel:
		
		for blk_sz in args.block_sizes:
			
			outputdir_string = check_folders(args.outputdir,args.cipher,blk_sz)
			
			"""
			Generate randomised pads for block size
			"""
			
			pads = [[randint(0,1) for i in range(blk_sz)] for j in range(args.keys)]
			
			"""
			# If block size is less than 23:
				# Generate the plain texts list of lists for total entropy size
				# Uses parallel processing w/ joblib to speed it up a bit
				# Product is a generator (?) returning:
					- all possible permutations for...
					- 0, 1 values...
					- with a total size of blk_size
			# Else:
				# the full entropy method might eat all your memory / CPU 
				# so do it using randint for the block size
			"""
			
			if blk_sz < 23:
				all_plains = parallel(delayed(give_list)(i) for i in product([0,1],repeat=blk_sz) )

				"""
				Product works in sequential order (0000, 0001, 0010 etc.) so we need to randomise the inputs
				Randomise the plain texts twice for two different variables to produce:
				- random order plain text list
				- random texts list
				"""

				all_plains, all_rands = plain_rands(all_plains,args.datasize)
				mtp_fullent_file_write(outputdir_string + 'plain/' + 'p_txt_bin.txt',all_plains)
				mtp_fullent_file_write(outputdir_string + 'plain/' + 'rand_bin.txt',all_rands)
				
			else:
				all_plains = list()
				for idx in range(args.datasize):
					
					plain = to_byte_fromint_subs( randint(0,pow(2,blk_sz)))
					file_write(outputdir_string + 'plain/' + 'p_txt_bin.txt',plain)
					all_plains.append( plain )
					
					rand = to_byte_fromint_subs( randint(0,pow(2,blk_sz)))
					file_write(outputdir_string + 'plain/' + 'rand_bin.txt',rand)
			
			print('plains, pads and rands generated and written to files in ' + outputdir_string)
			
			"""
			Delete random texts list to reduce memory usage
			"""
			
			del all_rands
			
			"""
			For each pad...
			"""

			for pad_idx,pad in enumerate(pads):
				
				file_write(outputdir_string + 'keys/' + 'keys.txt',''.join(map(str,pad)))
				
				cipher_print_message = 'cipher ' + s(pad_idx) + ' of ' + s(len(pads)) + ' of bit size ' + s(blk_sz)
				print(cipher_print_message  + ' starting...' )
				
				"""
				Generate ciphers from the pad and plain text lists of lists
				"""
				
				for plain in all_plains:
					c_txt = XOR_bits( plain , pad , blk_sz )
					file_write(outputdir_string + 'cipher/'+'ctxt_' + s(pad_idx) + '_bin.txt',''.join(map(str,c_txt)))
				print( cipher_print_message + ' generated & written')	
				
				# joblib version ?
				#ciphers = parallel( delayed(generate_bits)(all_plains,pads,bit_size,pad,j) for j in range(len(all_plains)) )
				
			print( 'all ciphers for ' + s(blk_sz) + ' generated...' )

################################################################################################################################################
# Subs script
################################################################################################################################################

	
def subs(args):
	
	SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' 

	substitutions = [{ SYMBOLS[i] : sample(SYMBOLS,len(SYMBOLS))[i] for i in range(0,len(SYMBOLS))} for key in range(args.keys) ]
	
	#### WRITE OUT?!
	
	print('\n --------------------------- \n+++ Substitutions being used: ')
	for sub_mapping in substitutions:
		print(sub_mapping)
		
	print('\n --------------------------- \n')

	for blk_sz in args.block_sizes:
		
		outputdir_string = check_folders(args.outputdir,args.cipher,blk_sz)
		
		write_count = 0
		
		while write_count < args.datasize:
			
			console.write("\r %s" % 'Block size : ' + s(blk_sz) + ' | ' + s(write_count*100/args.datasize)[0:5] \
			+ '% complete | count: ' +s(write_count)  )
			console.flush()
			
			plain_txt = rand_string_gen(blk_sz,SYMBOLS)
			
			file_write( outputdir_string + 'plain/' + 'p-txt-str.txt',plain_txt)
			file_write( outputdir_string + 'plain/' + 'p-txt-bin.txt', ''.join( to_byte_fromint_subs( to_byte( plain_txt ))) )
			
			rand_txt = rand_string_gen(blk_sz,SYMBOLS)
			
			file_write( outputdir_string + 'plain/' + 'rand-txt-str.txt',rand_txt)
			file_write( outputdir_string + 'plain/' + 'rand-txt-bin.txt',''.join( to_byte_fromint_subs( to_byte( rand_txt ))) )

			cipher_texts = [''.join( [ key[character] for character in plain_txt]) for key in substitutions ]
		
			for ctxt_idx,ctxt in enumerate(cipher_texts):
				file_write( outputdir_string + 'cipher/' + 'ctxt_' + s(ctxt_idx) + '_str.txt' , ctxt )
				file_write( outputdir_string + 'cipher/' + 'ctxt_' + s(ctxt_idx) + '_bin.txt',''.join(to_byte_fromint_subs(to_byte( ctxt ))) )
				
			write_count +=1

	print('\n ------------------------ \n+++ Completed! \n')

################################################################################################################################################
# Caesar script
################################################################################################################################################

def caesar(args):
	
	for blk_sz in args.block_sizes:
		
		outputdir_string = check_folders(args.outputdir,args.cipher,blk_sz)

		#key_choices = [randint(-25,25) for i in range(args.keys)]
		key_choices = [5,10,15,20,25]
		
		for key in key_choices:
			file_write( outputdir_string + 'keys/keys.txt' , to_byte_fromint_caesar(bin(key), blk_sz))
			
		"""
		Careful of bit flipping issues w/ caesar
		"""

		if min(key_choices) <0:
			min_val = min(key_choices) * (-1)
		else:
			min_val = 0
			
		if max(key_choices) <0:
			max_val = 0
		else:
			max_val = pow( 2 , blk_sz ) - max( key_choices )

		"""
		Generate random texts first, we can free up memory after then
		"""

		print('\n\ngenerating random texts\n\n')
		rands = caesar_plain_n_rand_gen(args.datasize,max_val,min_val,blk_sz)
		for rand in rands:
			file_write(outputdir_string + 'plain/rand-txt-bin.txt',rand)
		del rands

		"""
		# Generate plain + cipher texts and write c-txt files
			# caesar_plain_n_rand_gen outputs string values (using the to_byte_fromint function)
			# so we have to convert the plains back to binary using bin(int(x,2))
			# then convert back to binary string and write to file
		"""

		print('generating plain and cipher texts')
		plains = caesar_plain_n_rand_gen(args.datasize,max_val,min_val,blk_sz)
		for plain in plains:
			file_write(outputdir_string + 'plain/p-txt-bin.txt',plain)
			for key_idx,key in enumerate(key_choices):
				c_txt_bin = to_byte_fromint_caesar( bin( int( plain ,2 ) + key ) , blk_sz )
				file_write( outputdir_string + 'cipher/ctxt_' + s(key_idx) + '_bin.txt' , c_txt_bin )


## ARGPARSE

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('cipher', help="<REQUIRED> Which cipher are you generating for?",nargs=1,choices=['mtp','subs','caesar'],type=str)
	parser.add_argument('outputdir', help="<REQUIRED> DIRECTORY PATH (include '/') for output files",type=str,nargs=1)
	parser.add_argument('datasize', help="<REQUIRED> Max size of data set (n.b. mtp w/ block size less than 23 will be generated for full entropy )",type=int,nargs=1)
	parser.add_argument('block_sizes', help="<REQUIRED> block sizes (can pass multiple values)",nargs='+',type=int)
	
	parser.add_argument('--keys', help="<opt> number of keys to run for",type=int,default=4)
	
	args = parser.parse_args()
	
	args.datasize = args.datasize[0]
	args.cipher = args.cipher[0]
	args.outputdir = args.outputdir[0]
	
	print('\nARGUMENTS SELECTED:\n')
	print(args)
	print('\n')
	
	if args.cipher == 'mtp':
		mtp(args)
	elif args.cipher == 'subs':
		subs(args)
	elif args.cipher == 'caesar':
		caesar(args)