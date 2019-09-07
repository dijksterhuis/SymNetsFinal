
from Crypto.Cipher import DES
from Crypto.Random import get_random_bytes
from funx import *
from binascii import b2a_base64 as b2a
from binascii import a2b_base64 as a2b
from binascii import hexlify

numb_keys = 5
keys = [ get_random_bytes( 8 ) for j in range(0,numb_keys - 1) ]
key_idx = 0
for key in keys:
	key_idx +=1
	filepath = 'c-text-data/des/des_'+str(key_idx)+'_ciphertext_'
	keywrite(filepath,key,key_idx)
	#keywrite(filepath,[b2a(key).decode('ascii').rstrip('\n'),to_byte_fromint(key),hexlify(key) ,key_idx])
	for p_txt in open('p-text-data/8-block/p_txt_bin_10mill_nodupecheck.txt','r').readlines():
		cipher = DES.new( key , DES.MODE_ECB )
		c_txt = cipher.encrypt(p_txt.rstrip('\n '))
		filewrite(filepath,c_txt)
