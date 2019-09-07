                # VVVVVV

# !!! Don't forget to connect to the VPN !!!!

                 # ^^^^^


cd /Users/Mike/POSTGRAD/MSc\ Project/SymNets

sshfs dijksterhuis@rhea:/home/dijksterhuis/SymNets/neural-nets/tensorboard_design/results/ ./tensorboard-demo/mountpoint/ -o cache_timeout=5 -o cache_clean_interval=10 -o reconnect -o sync_readdir

tensorboard --logdir=./tensorboard-demo/mountpoint/

############################################

./neural-nets/tensorboard_design/run_models.py \
    ./data/plain/mtp20/p-txt-bin.txt \
    ./data/plain/mtp20/rand-bin.txt \
    ./data/cipher/mtp20/ \
    MTPdemo \
    P1 \
    --layers 1 \
    --width 40 \
    --lrate 0.7 \
    -r basic \
    -e 200 \
    --runs 1 \
    -d 1000000 \
    -i norm \
    --init_vars 0 1 \
    --hacts sigmoid\
    --yacts sigmoid\
    --optimiser GD \
    --loss cross_entropy

umount ./tensorboard-demo/mountpoint/