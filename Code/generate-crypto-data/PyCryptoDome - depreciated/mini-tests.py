from Crypto.Cipher import AES, DES, DES3, Blowfish
from Crypto.Random import get_random_bytes
from binascii import b2a_base64, a2b_base64, b2a_qp, a2b_qp
#from funx import to_byte_fromint, filewrite, keywrite

#def decrypt(in_txt,cipher_choice):#
#	plain_text_message = cipher_choice.encrypt(in_txt.rstrip('\n\r '))
#	return plain_text_message

##### variable definitions

c_texts, p_texts, key_file = dict(), dict(), list()
numb_to_check = 100
data_size = 150000

##### Decryption set up

method = input('(E)ncrypt or (D)ecrpyt: ')
cipher_choice = input('enter cipher choice: ')
key = a2b_base64(input('enter the key: '))
in_text = a2b_base64(str(input('enter the input text: ')))
print(input('try no str input'))
print(str(input('try str input')))
print('key: ' + str(key))
print('key_len: '+ str(len(key))+'\n--------------\n')
print('input txt: ' + str(in_text))
print('input_len: '+ str(len(in_text))+'\n--------------\n')

if cipher_choice =='AES':
	cipher = AES.new( key , AES.MODE_ECB )
elif cipher_choice == 'DES':
	cipher = DES.new( key , DES.MODE_ECB )
elif cipher_choice == 'DES3':
	cipher = DES3.new( key , DES3.MODE_ECB )
elif cipher_choice == 'BF':
	cipher = Blowfish.new( key , Blowfish.MODE_ECB )
if method in ['D','d']:
	out_text = cipher.decrypt(in_text)
elif method in ['E','e']:
	out_text = cipher.encrypt(in_text)

print(out_text.decode('ascii'))
print('\n------------------\n')
