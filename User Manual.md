# User Manual

All instructions in this manual require a bash terminal session.

It is assumed that you will be working from the `./SymNets` directory on the USB disk this manaul exists on.
If you aren't you'll need to edit the commands below.

## Basic installations

To run the programs in this folder, Python 3 needs to be installed.
For instructions on how to install Python, please refer to the Hitchiker's Guide to Python (http://docs.python-guide.org/en/latest/starting/installation/).

Setuptools & Pip will be required, so please ensure they are installed.

## Further installations

If Python and Pip are installed on your system, please also ensure you have Virtualenv installed (https://virtualenv.pypa.io/en/stable/installation/).
```bash
pip install --upgrade virtualenv
```
### Tensorflow

To install Tensorflow, run the following commands:

```bash
virtualenv Tensorflow
source ./Tensorflow/bin/activate
pip install --upgrade tensorflow
deactivate
```

### PyCryptoDome - Depreciated!

PyCryptoDome was used in the early stages of the project, but was dropped from usage.
It is included here purely for reference. Python 3.4 must be installed on your system.

To install PyCryptoDome run the following commands:
```bash
virtualenv --python=/usr/bin/python3.4 PyCryptoDome
source ./PyCryptoDome/bin/activate
pip install --upgrade PyCryptoDome
deactivate
```
## Generating Data

### CipherGen

Navigate to the path where CipherGen.py is stored.
CipherGen.py takes the following arguments:

```
cipher_choice {mtp,subs,caesar}
outputdirectory {path/to/outputfolder/}
datasize {integer value - 1000}
block_sizes {allows for multiple entries seperated by space, integer value - 8}
```

Run the following command from the SymNets folder to generate a dataset of 1000 data points for an MTP cipher of block size 20:

```bash
mkdir ./testdata/mtp-test-dir
python ./Code/generate-crypto-data/CipherGen.py mtp ./testdata/mtp-test-dir 1000 20
```

### CryptoPlainGen

Navigate to the path where CipherGen.py is stored.
CipherGen.py takes the following arguments:

```
file_type {plain,rand}
outputdirectory {path/to/outputfolder/}
datasize {integer value - 1000}
block_sizes {allows for multiple entries seperated by space, integer values - 8}
```

Run the following commands from the SymNets folder to generate a dataset of 1000 data points for plain texts of block size 8:

```bash
mkdir ./testdata/plains
python ./Code/generate-crypto-data/CryptoPlainGen.py plain ./testdata/plains 1000 8
```

Run the following commands from the SymNets folder to generate a dataset of 1000 data points for random texts of block size 8:

```bash
mkdir ./testdata/rands
python ./Code/generate-crypto-data/CryptoPlainGen.py rand ./testdata/rands 1000 8
```

### DES

Keys must be generated before being used. There are some examples keys in the keys folder.
Navigate to the path where CipherGen.py is stored.
CipherGen.py takes the following arguments:

```
plain text input {path/to/file.txt}
outputdir {path/to/outputfolder/}
full or single run? {full,single}
datasize {integer value - 1000}
```

Run the following commands from the SymNets folder to generate a dataset of 1000 data points for DES:

```bash
python ./Code/generate-crypto-data/DES.py \
  ./testdata/plains/crypto8blk/plain/p_txt_bin.txt \
    ./testdata/DESoutputdir/ \
      full 100 \
      --rounds 3 \
      --key_file ./Code/generate-crypto-data/DES_keys.txt
```

## Running a Network

### Tensorflow

This is how to run a basic neural network. Other modes won't be covered by this manual.
To find out about other network run options, see commentary in the Run_models.py file.

#### Example 1 - MTP C1

Start a Tensorflow virtual environment and run an example network for the MTP test data we generated earlier:

```bash
source ./Tensorflow/bin/activate ; \
python ./Code/neural-nets/tensorboard-design/run_models.py \
  ./testdata/mtp-test-dir/plain/mtp20/p-txt-bin.txt \
  ./testdata/mtp-test-dir/plain/mtp20/rand-bin.txt \
  ./testdata/mtp-test-dir/cipher/mtp20/ \
  MTPdemoC1 \
  C1 \
  --layers 1 \
  --width 10 \
  --lrate 0.7 \
  -r basic \
  -e 200 \
  -d 1000 \
; deactivate
```

#### Example 2 - DES C2

Start a Tensorflow virtual environment and run an example network for the DES test data we generated earlier:

```bash
source ./Tensorflow/bin/activate ; \
python ./Code/neural-nets/tensorboard-design/run_models.py \
  ./testdata/plains/crypto8blk/plain/p_txt_bin.txt \
  ./testdata/rands/crypto8blk/plain/rand_bin.txt \
  ./testdata/DESoutputdir/c-txts/ \
  DESdemoC2 \
  C1 \
  --layers 1 \
  --width 10 \
  --lrate 0.7 \
  -r basic \
  -e 200 \
  -d 1000 \
; deactivate
```

### Tensorboard

Both of the example Tensorflow runs can be visualised with Tensorboard.
Run one of the commands below, depending on which Tf examples you have run.
Then navigate to localhost in a webbrowser to view the Tensorbaord summary.

```bash
# For example 1:
source ./Tensorflow/bin/activate ; \
  tensorboard --logdir ./Code/neural-nets/tensorboard_design/results/MTPdemoC1/
# For example 2:
source ./Tensorflow/bin/activate ; \
  tensorboard --logdir ./Code/neural-nets/tensorboard_design/results/DESdemoC2/

# Make sure to deactivate your virtual environment!
deactivate
```