echo ' THIS SCRIPT WILL REMOVE ALL EXISTING DATA SETS'
echo " Comment out relevant lines if you don't want this to happen"
echo '---------------------------------------------------------------'
echo ' '

echo 'removing plaintexts'
sudo rm -f ~/Documents/SymNets/data-gen/p-text-data/16-block/*.txt
sudo rm -f ~/Documents/SymNets/data-gen/p-text-data/8-block/*.txt
#sudo rm -f ~/Documents/SymNets/data-gen/p-text-data/casaer/*.txt
echo 'removed plaintexts'
echo 'removing cipher texts'
#sudo rm -f ~/Documents/SymNets/data-gen/c-text-data/casaer/*.txt
sudo rm -f ~/Documents/SymNets/data-gen/c-text-data/des/*.txt
sudo rm -f ~/Documents/SymNets/data-gen/c-text-data/3des/*.txt
sudo rm -f ~/Documents/SymNets/data-gen/c-text-data/aes/*.txt
sudo rm -f ~/Documents/SymNets/data-gen/c-text-data/blowfish/*.txt
echo 'removed cipher texts'

#echo ' creating caeser plain texts'
#python3 ~/Documents/SymNets/data-gen/py-scripts/casaer_plain_and_rand_files.py
#echo ' '
#echo ' casaer cipher done.'
echo '---------------------------------------------------------------'
echo ' '
echo ' generating crypto plain texts '
python3 ~/Documents/SymNets/data-gen/py-scripts/plain-text-blocklen.py
echo ' '
echo 'plain texts generated'
echo ' '
#echo ' creating casaer ciphers'
#python3 ~/Documents/SymNets/data-gen/py-scripts/casaer.py
#echo ' casaer ciphers generated'
#echo ' '
docker run -it --rm --net=host -v ~/Documents/SymNets/data-gen:/home/ -w '/home' --name des pycrypto:live /bin/ash -c 'python3 py-scripts/des.py'
echo 'DES run'
docker run -it --rm --net=host -v ~/Documents/SymNets/data-gen:/home/ -w '/home' --name 3des pycrypto:live /bin/ash -c 'python3 py-scripts/3des.py'
echo '3DES run'
docker run -it --rm --net=host -v ~/Documents/SymNets/data-gen:/home/ -w '/home' --name aes pycrypto:live /bin/ash -c 'python3 py-scripts/aesv2.py'
echo 'AES run'
docker run -it --rm --net=host -v ~/Documents/SymNets/data-gen:/home/ -w '/home' --name bfish pycrypto:live /bin/ash -c 'python3 py-scripts/blowfish.py'
echo 'BF run'
echo ' '
echo '---------------------------------------------------------------'
echo 'cipher texts generated'
