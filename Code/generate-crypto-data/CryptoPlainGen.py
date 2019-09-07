#!/usr/bin/python

from os import path, makedirs as mkdir
from datetime import datetime as dt
from random import randint
from sys import stdout as console
import argparse

def folder(outputdir,blk_sz):
	
	"""
	##########################################################
	Function to check if output folders exit, creating if not
	##########################################################
	"""
	
	outputdir_string = outputdir + 'crypto' + s(blk_sz) + 'blk/'
	if not path.exists(outputdir_string):
		mkdir(outputdir_string + 'plain/')
	return outputdir_string + 'plain/'

def s(string):

	"""
	#########################
	Function to simplify code
	#########################
	"""
	
	return str(string)

def to_byte_fromint(string):
	
	"""
	#######################################################
	Function to convert strings into binary representations
	#######################################################
	
	In pseudocode:
	
	1. Convert the utf-8 encoded string	into a list of binary
	2. Split the binary strings if they have 0b in them (python byte representation)
	3. Add padding on for the removed parity bit + any missing bits
	
	"""
	
	byte_vals = ''.join(map(bin,bytearray(string,'utf8'))).split(' ')
	split_byte = byte_vals[0].split('0b')
	padded_byte = ''.join( [''.join(['0' for i in range(0,8 - len(j))]) + j for j in split_byte[1:len(split_byte)]])
	return padded_byte

def console_print(blk_sz,counter,datasize,starttime):
	
	"""
	##########################################################
	Function to print usage stats to bash (sys.stdout) console
	##########################################################
	"""
	
	console.write( "\r %s" % 'Block size : ' + str(blk_sz) + ' | ' + str((counter/datasize)*100)[0:5] + '% complete' \
	+ ' | Time elapsed: ' + str(dt.now() - starttime)[0:7] + " | Current write count: " + str(counter) + " | Total to write: " + str(datasize) \
	)
	console.flush()

def end_print(counter,blk_sz):

	"""
	##############################################
	Function to print a final confirmation message
	##############################################
	"""
	
	print('\n-----------------------------------------------------------------\n')
	print(str(counter) + ' total writes for block size ' + str(blk_sz))
	print('\n-----------------------------------------------------------------\n')

def plains(args):
	
	"""
	################################################################################
	Function to generate the plain text messages for the block cipher crypto systems
	################################################################################
	
	In pseudocode:

	1. for each block size
		2. if the data set size limited hasn't been reached yet
			3a. generate a list (with length = block size ) of indexes for the SYMBOL string variable 
			3b. get the corresponding SYMBOLs in a list
			3c. join the string elements of the SYMBOLS list into a single string, creating the plain text
			3d. Write string and binary representations of the plain text to a file
			3e. increment the data size counter

	################################################################################
	
	# For plain texts, we want to use a limited set of ASCII symbols
		# Reason 1:
			# Should give networks an advantage
			# ASCII symbols will have a 0 parity bit at the 0 position
			# This reduces the random distribution of input bit values to the network
			# Using random binary values would remove this useful pattern
		# Reason 2:
			# Originally, PyCrpytoDome was used to generate Crypto Cipher texts
			# Pycryptodome takes strings as inputs and converts to hex (byte arrays)
	"""
	
	SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
	
	print("\n CREATING PLAIN TEXTS \n")
	for blk_sz in args.block_size:
		outpath = folder(args.outputdir,blk_sz)
		counter, starttime = 0 , dt.now()
		while counter < args.datasize:
			console_print(blk_sz,counter,args.datasize,starttime)
			plain_text = ''.join([SYMBOLS[idx] for idx in [ randint(0,len(SYMBOLS)-1) for i in range(0,blk_sz) ] ])
			with open( outpath + 'p_txt_bin.txt','a') as f:
				f.write( ''.join( to_byte_fromint( plain_text )) + '\n')
			with open( outpath + 'p_txt_str.txt','a') as f:
				f.write( ''.join( plain_text ) + '\n')
			counter += 1
		end_print(counter,blk_sz)

def rand(args):
	
	"""
	#######################################################################################
	Function to generate the rand text messages for the block cipher crypto systems C1 task
	######################################################################################
	
	In pseudocode:

	1. for each block size
		2. if the data set size limited hasn't been reached yet
			3a. generate a list (with length = block size * 8) random 1 or 0 values
			3b. convert the list elements list to string then join them all together
			3c. Write binary representations of the rand text to a file
			3e. increment the data size counter
	
	######################################################################################
	
	# For random texts, we don't want to use ASCII string because of the parity bits
	# It allowed the netwroks to classify too easily
	# They only needed to pick up whether there was a 0 parity bit or not to solve the C1 problem
	# I.e. it was classifying if there was a parity bit in the random text data or not
	# Rather than classifying if the data was encrypted or not
	
	"""
	print("\n CREATING RANDOM TEXTS \n")
	for blk_sz in args.block_size:
		outpath = folder(args.outputdir,blk_sz)
		counter, starttime = 0 , dt.now()
		while counter < args.datasize:
			console_print(blk_sz,counter,args.datasize,starttime)
			rand_bits = ''.join([str(randint(0,1)) for i in range (0,blk_sz*8)])
			with open(outpath + 'rand_bin.txt','a') as f:
				f.write( rand_bits + '\n' )
			counter +=1
		end_print(counter,blk_sz)

if __name__ == '__main__':

	"""
	##########################################################
	argparse argument declarations
	##########################################################
	"""
	
	parser = argparse.ArgumentParser()
	parser.add_argument('text', help="<REQUIRED> which text to generate - plain or rand",choices=['plain','rand'],nargs=1,type=str)
	parser.add_argument('outputdir', help="<REQUIRED> filepath for output file, 'crypto[block size]blk' folder automatically added to end of path",nargs=1,type=str)
	parser.add_argument('datasize', help="<REQUIRED> number of lines to generate",nargs=1,type=int)
	parser.add_argument('block_size', help="<REQUIRED> block size",nargs='+',type=int)
	
	args = parser.parse_args()
	
	"""
	##########################################################
	argparse variable modifications
	##########################################################
	"""
	
	args.text = args.text[0]
	args.outputdir = args.outputdir[0].rstrip('/') + '/'
	args.datasize = args.datasize[0]
	
	if args.text == 'plain':
		plains(args)
	elif args.text == 'rand':
		rand(args)