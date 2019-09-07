

from Crypto.Cipher import Blowfish as bf
from Crypto.Random import get_random_bytes
from funx import *
from binascii import b2a_base64 as b2a
from binascii import a2b_base64 as a2b


key_sizes = [32, 64 , 96 , 128 , 168, 192 , 256]
numb_keys = 5

key_vals = { i : [ get_random_bytes(int( i/8 )) \
for j in range(0,numb_keys - 1)] for i in key_sizes }

for key_size in key_vals:
	key_idx = 0
	for key in key_vals[key_size]:
		print('key idx: '+str(key_idx)+' | key size: '+str(key_size))
		key_idx+=1
		filepath = 'c-text-data/blowfish/bf'+str(key_size)+'_key_'+str(key_idx)+'_ciphertext_'
		keywrite(filepath,key,key_idx)
		for p_txt in open('p-text-data/16-block/p_txt_bin_10mill_nodupecheck.txt','r').readlines():
			cipher = bf.new( key , bf.MODE_ECB )
			c_txt = cipher.encrypt(p_txt.rstrip('\n '))
			filewrite(filepath,c_txt)
