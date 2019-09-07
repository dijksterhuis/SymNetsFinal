s
from Crypto.Cipher import DES3
from Crypto.Random import get_random_bytes
from funx import *
from binascii import b2a_base64 as b2a
from binascii import a2b_base64 as a2b


numb_keys = 5
keys = [ get_random_bytes( 16 ) for j in range(0,numb_keys - 1)]

key_idx = 0

for key in keys:
	key_idx+=1
	filepath = 'c-text-data/3des/3des_'+str(key_idx)+'_ciphertext_'
	keywrite(filepath,[b2a(key).decode('ascii'),to_byte_fromint(key),key_idx])
	for p_txt in open('p-text-data/8-block/p_txt_str.txt','r').readlines():
		cipher = DES3.new( key , DES3.MODE_ECB )
		c_txt = cipher.encrypt(p_txt.rstrip('\n '))
		filewrite(filepath,c_txt)
