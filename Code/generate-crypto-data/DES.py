#!/usr/bin/python

from sys import argv
from sys import stdout as console
import argparse
import os

#https://gist.github.com/eigenein/1275094

def leni(data):
	"""
	function to return the last index value, rather than length
	"""
	return len(data) - 1

def single_filewrite(data,outputdir,thingtorun):
	"""
	function for oneline writing data to a file
	"""
	with open(outputdir + 'output.txt','a') as f:
		f.write(data + '\n')

def single(ptxt_file,key_file,outputdir,thingtorun,datasize):
	"""
	Generate cipher text data for a single component of DES only
	
	In pseudocode:
		1. Open plain text file
		2. for each line in the plain text file
			3a. if not reached the data set size limit yet
				4a. process it with the required DES component
				4b. write output to specified file
			3b. otherwise break the loop (end the function)
	
	"""
	with open(ptxt_file,'r') as f_in:
		print('\n')
		for ptxt_idx,line in enumerate(f_in.readlines()):
			if ptxt_idx < datasize:
				console.write('\r %s' % str(ptxt_idx+1) + ' %' + 'P_TEXT lines')
				console.flush()

				line = line.rstrip('\n\r')
				# 6 blocks in
				# 4 blocks out
				if thingtorun == 'sbox':
					output = sboxing(line)
					single_filewrite(output,outputdir,thingtorun)
				# 8 blocks in
				# 8 blocks out
				if thingtorun == 'initperm64':
					output = initial_perm64(line)
					single_filewrite(output,outputdir,thingtorun)
				# 4 blocks in
				# 6 blocks out
				if thingtorun == 'expand':
					output = Xpansion(line)
					single_filewrite(output,outputdir,thingtorun)
				# 4 blocks in
				# 4 blocks out
				if thingtorun == 'perm32':
					output = perm32(line)
					single_filewrite(output,outputdir,thingtorun)
				# 8 blocks in
				# 8 blocks out
				if thingtorun == 'ipminusone':
					output = IPminus1(line)
					single_filewrite(output,outputdir,thingtorun)
			else:
				break


def full_run(ptxt_file,key_file,outputdir,datasize,rounds):
	
	"""
	PYTHON IMPLEMENTATION OF DES
	
	In pseudocode:
		1. check if required directories exit, creating them if not
		2. read in the keys from a txt file
		3. for each key
			4a. generate the n_rounds x subkeys
			4b. write all the subkeys to a file
			4c. for each line on the plain text file
				5. if the data set size limit hasn't been reached
					6a. do each DES operation (including rounds)
					6b. write the output of DES operation (including rounds)
			4d. write the cipher text output to file
	
	"""
	
	folders = [ outputdir+'/round'+str(i) for i in range(rounds)]
	folders.append(outputdir+'/inits')
	folders.append(outputdir+'/c-txts')
	for folder in folders:
		if not os.path.exists(folder):
			os.makedirs(folder)
	del folders
	
	# get key input file and parse data
	#### MUST ONLY BWE BIT STRING INPUTS
	#### remember the hex values can have spaces
	with open(key_file,'r') as f:
		keys = [i.rstrip('\n\r') for i in f.readlines()]

	# create all the subkeys, write to file
	for key_idx,key in enumerate(keys):
		subkeys = keyschedule(key)
		for subkey in subkeys:
			with open(outputdir + 'subkeys_key' + str(key_idx)+'.txt','a') as f:
				f.write(subkey + '\n')

		# get the p txt data, perform DES encryption line by line
		with open(ptxt_file,'r') as f_in:
			print('\n')
			for ptxt_idx,line in enumerate(f_in.readlines()):
				if ptxt_idx < datasize:
					console.write('\r %s' % str(ptxt_idx+1) + ' %' + 'P_TEXT lines proc for key: '+str(key_idx+1))
					console.flush()

					# 64 bits to 64 bits permuated
					p_txt_p64d = initial_perm64(line.rstrip('\n\r'))
					with open(outputdir + 'inits/'+'permsiddyfour_key'+str(key_idx)+'.txt','a') as f:
						f.write(p_txt_p64d + '\n')

					# 64 bits to two 32 bit pairs
					left , right_nmin1 = init_split(p_txt_p64d)
					with open(outputdir + 'inits/' +'split_key' + str(key_idx)+'.txt','a') as f:
						f.write(p_txt_p64d + '\n')

					# perform the rounds
					for DESround in range(rounds):

						# right only processed in feistal

						# expand 32 bit RHS to 48 bits
						right = Xpansion(right_nmin1)
						with open(outputdir +'/round'+str(DESround)+'/' + 'xpansion_key' + str(key_idx)+'.txt','a') as f:
							f.write(right + '\n')

						# XOR 48 bits with 48 bit subkey
						right = f_XOR(right,subkeys[DESround])
						with open(outputdir +'/round'+str(DESround)+'/' + 'fXOR_key' + str(key_idx)+'.txt','a') as f:
							f.write(right + '\n')

						# SBOX 48 bits to 32 bits
						right = sboxing(right)
						with open(outputdir +'/round'+str(DESround)+'/' + 'sbox_key' + str(key_idx)+'.txt','a') as f:
							f.write(right + '\n')

						# Permuate 32 bits to 32 bits
						right = perm32(right)
						with open(outputdir +'/round'+str(DESround)+'/' + 'permtirtytwo_key' + str(key_idx)+'.txt','a') as f:
							f.write(right + '\n')

						# assign values for next round
						leftnext, rightnext = right_nmin1, f_XOR(left,right)
						left, right_nmin1 = leftnext, rightnext

					# final 64 bit to 64 bits permutation
					output = IPminus1(rightnext + leftnext)

					with open(outputdir +'/c-txts' + 'c_txt_bin_key' + str(key_idx)+'.txt','a') as f:
						f.write(output + '\n')
				else:
					break

###############################################################
### Global DES funx

def initial_perm64(data):
	"""
	Function for the IP64 table
	
	1. For 64 bit positions
	2. find the coresponding bit from the input data
	3. according to the table
	4. output as a string
	"""
	print(data)
	init_perm64 = [57, 49, 41, 33, 25, 17, 9,  1,
			59, 51, 43, 35, 27, 19, 11, 3,
			61, 53, 45, 37, 29, 21, 13, 5,
			63, 55, 47, 39, 31, 23, 15, 7,
			56, 48, 40, 32, 24, 16, 8,  0,
			58, 50, 42, 34, 26, 18, 10, 2,
			60, 52, 44, 36, 28, 20, 12, 4,
			62, 54, 46, 38, 30, 22, 14, 6]
	output_bits = [ data[init_perm64[i]] for i in range(64) ]
	return ''.join(output_bits)

def init_split(data):
	"""
	Do the left / right DES round splitting
	"""
	return data[0:32], data[32:64]

def IPminus1(data):
	"""
	Function for the IP-1 table
	
	1. For 64 bit positions
	2. find the coresponding bit from the input data
	3. according to the table
	4. output as a string
	"""
	IPtable2 = [39,  7, 47, 15, 55, 23, 63, 31,
			38,  6, 46, 14, 54, 22, 62, 30,
			37,  5, 45, 13, 53, 21, 61, 29,
			36,  4, 44, 12, 52, 20, 60, 28,
			35,  3, 43, 11, 51, 19, 59, 27,
			34,  2, 42, 10, 50, 18, 58, 26,
			33,  1, 41,  9, 49, 17, 57, 25,
			32,  0, 40,  8, 48, 16, 56, 24]

	output_bits = [ data[IPtable2[i]] for i in range(64) ]
	return ''.join(output_bits)

###############################################################
### FEISTAL FUNCTION

def Xpansion(data):
	"""
	### EXPANSION
	# the 32-bit half-block is expanded to 48 bits using the expansion permutation
	# by duplicating half of the bits.
	# The output consists of eight 6-bit (8 * 6 = 48 bits) pieces
	# each containing a copy of 4 corresponding input bits
	# plus a copy of the immediately adjacent bit from each of the input pieces to either side.
	"""

	expansion_table = [\
		31,  0,  1,  2,  3,  4,\
		 3,  4,  5,  6,  7,  8,\
		 7,  8,  9, 10, 11, 12,\
		11, 12, 13, 14, 15, 16,\
		15, 16, 17, 18, 19, 20,\
		19, 20, 21, 22, 23, 24,\
		23, 24, 25, 26, 27, 28,\
		27, 28, 29, 30, 31,  0 ]

	if len(data) != 32 and type is not str:
		print('received either a bit string of incorrect length, or not a string, exiting')
		pass

	input_bits = ''.join([ data[i:i+4] for i in range(0,32,4) ])
	output_bits = [ input_bits[expansion_table[i]] for i in range(len(expansion_table)) ]
	return ''.join(output_bits)

def sboxing(data):
	# from https://gist.github.com/eigenein/1275094
	
	"""
	Function for the SBox
	
	1. Check input length and type
	2. split input bits into blocks of 6
	3. for each SBox
	4. lookup the corresponding output value based on the 6 input bits
	5. add padding if required
	"""
	
	sboxes = [
	# S1
	[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
	0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
	 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
	 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

	# S2
	[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
	 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
	 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
	 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

	# S3
	[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
	 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
	 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
	 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

	# S4
	[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
	 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
	 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
	 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

	# S5
	[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
	 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
	 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
	 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

	# S6
	[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
	 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
	 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
	 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

	# S7
	[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
	 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
	 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
	 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

	# S8
	[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
	 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
	 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
	 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
	]

	if len(data) != 48 and type is not str:
		print('received either a bit string of incorrect length, or not a string, exiting')
		exit()
	else:

		input_bits = [ data[i:i+6] for i in range(0,48,6) ]
		result = list()

		# per sbox (per 6 bits in data)
		for i in range(8):

			address_1 = (int(input_bits[i][0:6:5],2) * 16)
			address_2 = (int(input_bits[i][1:5],2))

			result.append( str(bin(sboxes[i][address_1 + address_2]))[2:6] )

			#print(address_1,address_2,result[i],  ('0' * (4 - len(result[i]))) + result[i])
			#[ str(bin(sboxes[i][ (int(input_bits[i][0:6:5],2) * 16) + (int(input_bits[i][1:5],2)) ]))[2:6] for i in range(8)]

		output_bits = [('0' * (4 - len(s))) + s for s in result]

		return ''.join(output_bits)

def f_XOR(s1,s2):
	# https://stackoverflow.com/a/2612730
	# convert strings to a list of character pair tuples
	# go through each tuple, converting them to ASCII code (ord)
	# perform exclusive or on the ASCII code
	# then convert the result back to ASCII (chr)
	# merge the resulting array of characters as a string
	return ''.join(str(ord(a) ^ ord(b)) for a,b in zip(s1,s2))


def perm32(data):
	"""
	final feistel perm32 table function
	
	1. For 32 bit positions
	2. find the coresponding bit from the input data
	3. according to the table
	4. output as a string
	"""
	
	final_perm32 = [15, 6, 19, 20, 28, 11, 27, 16, 0, 14, 22, 25, 4, 17, \
	30, 9, 1, 7, 23,13, 31, 26, 2, 8, 18, 12, 29, 5, 21, 10, 3, 24]

	output_bits = [ data[final_perm32[i]] for i in range(32) ]
	return ''.join(output_bits)

###############################################################
### KEY ROUND GENERATION

def keyschedule(input_key):
	"""
	# Uses below functions to generate 16 sub keys for rounds
	# surprisingly easy!
	### IMPORTANT NOTE!
	# DES only uses the 16 GENERATED sub keys - not the key in posn 0
	"""
	#with open(key_in_f,'r') as f_in:
	#	input_key = f_in.read().rstrip('\n\r')
	shifted_keys = list()
	shifted_keys.append(key_split(PC1(input_key)))
	for i in range(16):
		left_key = key_shift( shifted_keys[i][0] , i )
		right_key = key_shift( shifted_keys[i][1] , i )
		shifted_keys.append( [ left_key , right_key ])
	permd_keys = shifted_keys
	for i in range(17):
		permd_keys[i] = PC2( shifted_keys[i][0] + shifted_keys[i][1] )
	return permd_keys[1:17]

def PC1(data):
	"""
	PC1 table function
	
	1. For 56 bit positions
	2. find the coresponding bit from the input data
	3. according to the table
	4. output as a string
	"""
	pc1_table = [56, 48, 40, 32, 24, 16,  8,
			  0, 57, 49, 41, 33, 25, 17,
			  9,  1, 58, 50, 42, 34, 26,
			 18, 10,  2, 59, 51, 43, 35,
			 62, 54, 46, 38, 30, 22, 14,
			  6, 61, 53, 45, 37, 29, 21,
			 13,  5, 60, 52, 44, 36, 28,
			 20, 12,  4, 27, 19, 11,  3]

	output_bits = [ data[pc1_table[i]] for i in range(56) ]
	return ''.join(output_bits)

def key_split(data):
	"""
	Do the left / right key splitting
	"""
	left, right = data[0:28], data[28:56]
	return left, right

def key_shift(data,key_idx):
	"""
	Shift key bit values by x for respective rounds
	
	1. select the number of bit shifts based on the current round number
	2. bits on LHS of the shift index value go to the RHS
	3. bits on RHS of the shift index value go to the LHS
	
	"""
	
	left_rotations = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
	shift = left_rotations[key_idx]
	flipped_bits = data[0:shift]
	shifted_bits = data[shift:28]
	return shifted_bits + flipped_bits

def PC2(data):
	"""
	PC2 table function
	
	1. For 56 bit positions
	2. find the coresponding bit from the input data
	3. according to the table
	4. output as a string
	"""
	pc2_table = [
			13, 16, 10, 23,  0,  4,
			 2, 27, 14,  5, 20,  9,
			22, 18, 11,  3, 25,  7,
			15,  6, 26, 19, 12,  1,
			40, 51, 30, 36, 46, 54,
			29, 39, 50, 44, 32, 47,
			43, 48, 38, 55, 33, 52,
			45, 41, 49, 35, 28, 31]
	output_bits = [ data[pc2_table[i]] for i in range(48) ]
	return ''.join(output_bits)


###############################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	file_args = parser.add_argument_group('Files and Directories','Arguments for passing file/directory paths')
	runtime_args = parser.add_argument_group('Run time properties','Arguments for how to run this script')
	
	file_args.add_argument('ptxt_file', help="<REQUIRED> filepath for ptxt input",type=str)
	file_args.add_argument('--key_file', help="filepath for key input, <REQUIRED> for FULL runs",type=str)
	file_args.add_argument('outputdir', help="<REQUIRED> directory for output file(s) to be written to",type=str)
	
	runtime_args.add_argument('run_type', help="<REQUIRED> what type of priocesing you're doing",choices=['full','single'],type=str)
	runtime_args.add_argument('--thingtodo', help="which DES component to run for, <REQUIRED> for SINGLE runs",choices=['sbox','initperm64','expand','perm32','ipminusone'],type=str)
	runtime_args.add_argument('datasize', help="<REQUIRED> How many lines of file you want processed",type=int)
	runtime_args.add_argument('--rounds', help="How many DES rounds to perfrom, <REQUIRED> for FULL runs",type=int)
	
	args = parser.parse_args()
	
	if args.run_type == 'full':
		full_run(args.ptxt_file,args.key_file,args.outputdir,args.datasize,args.rounds)

	elif args.run_type == 'single':
		single(args.ptxt_file,args.key_file,args.outputdir,args.thingtodo,args.datasize)

	print('\n Processing Completed... \n\n')
