#!/usr/bin/python
from sys import stdout as console
from DES_simplified import f_XOR, Xpansion, init_split
import argparse

def main(args):
	"""
	################################################
	Function to run the simplified rounds experiment
	################################################

	Uses functions from the DES_simplified file for the most part
	
	################################################
	
	In pseudocode:
	
		1. for each plain text
			2. split into left(0) + right(0)
			3. for each round (of 16)
				4a. left output = Xpanded right input
				4b. right output = XOR of left and right inputs
				5. write the outputs to a file
				6. next round's inputs are the current outputs
	 
	"""
	
	with open(args.infile,'r') as f:
		data = [i.rstrip('\r\n') for i in f.readlines()]
		
	for idx,string in enumerate(data):
		
		input_left, input_right = init_split(string)
		
		for rnd in range(16):
			
			# Xpansion chosen as it's nice and simple, but we could choose anything, even the caesar shift
			
			output_left, output_right = Xpansion(input_right), f_XOR(input_left,input_right)
			
			with open(args.outfile + 'round' + str(rnd) + '.txt','a') as f:
				f.write(output_left+output_right + '\n')
				
			input_left, input_right = output_left, output_right
			
			console.write("\r %s" % 'processed lines: ' + str(idx) +'| round: '+str(rnd)+ ' | output string: ' + output_left+output_right )
			console.flush()
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('infile', help="<REQUIRED> filepath for input file",nargs=1,type=str)
	parser.add_argument('outfile', help="<REQUIRED> filepath for output file",nargs=1,type=str)
	args = parser.parse_args()
	args.outfile, args.infile = args.outfile[0], args.infile[0]
	main(args)