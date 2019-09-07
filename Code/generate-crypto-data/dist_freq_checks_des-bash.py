#!/usr/bin/python

import sys

def create_dict(listy):
	freqs = dict()
	for entry in listy:
		#print(entry)
		for i in range(0,len(entry)):
			if entry[i] == 'b':
				pass
			#print(i,entry[i])
			elif i not in freqs.keys() and int(entry[i]) is 1:
				freqs[i] = 1
			elif int(entry[i]) is 1:
				freqs[i] += 1
	return freqs


def key_check(k,d):
	if k not in d.keys():
		return 0
	else:
		return d[k]

def sorted_dict_print(dict1,dict2,dict3):
	print('--------------------------')
	print('Total 1 bits per key:')
	for i in range( max([len(d) for d in [dict1,dict2,dict3]]) ):
		print(i+1, key_check(i,dict1) , key_check(i,dict2) , key_check(i,dict3) )

def get_data(filepath):
	print('Getting data for '+str(filepath))
	with open(filepath,'r') as f:
		data = [line.rstrip('\n') for line in f.readlines()]
	return data

def main(sysargs):
	data1 , data2 , data3 = get_data(sysargs[1]) , get_data(sysargs[2]) , get_data(sysargs[3])
	freqs1, freqs2, freqs3 = create_dict(data1), create_dict(data2), create_dict(data3)
	print('--------------------------\nLocations with 1 bits:')
	print(sorted(freqs1.keys()))
	print(sorted(freqs2.keys()))
	print(sorted(freqs3.keys()))
	sorted_dict_print(freqs1, freqs2, freqs3)

if __name__ == '__main__':
	main(sys.argv)
