

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from funx import *
import sys
from binascii import b2a_base64 as b2a
from binascii import a2b_base64 as a2b


in_filepath = 'p-text-data/16-block/p_txt_str.txt'

key_sizes = [128 , 192 , 256]
numb_keys = 5

key_vals = { i :\
 [ get_random_bytes(int( i/8 )) \
 for j in range(0,numb_keys - 1)] for i in key_sizes }


for key_size in key_vals:
	key_idx = 0
	for key in key_vals[key_size]:
		key_idx+=1
		filepath = 'c-text-data/aes/aes'+str(key_size)+'_key_'+str(key_idx)+'_ciphertext_'
		keywrite(filepath,[b2a(key).decode('ascii'),to_byte_fromint(key),key_idx,key_size])
		for p_txt in open(in_filepath,'r').readlines():
			cipher = AES.new( key , AES.MODE_ECB )
			c_txt = cipher.encrypt(p_txt.rstrip('\n '))
			filewrite(filepath,c_txt)
