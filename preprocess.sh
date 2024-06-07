#!/bin/bash

# Prepare paired cluster label (K-means on log Mel feature) for training 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Z4WU6m5v1Aq8MpzpoYIggBcLD-SQUkr9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Z4WU6m5v1Aq8MpzpoYIggBcLD-SQUkr9" \
     -O libri-360.tar.gz && rm -rf /tmp/cookies.txt
tar -xvf libri-360.tar.gz 
rm libri-360.tar.gz 

# Extracting the data and the cluster 
python3 preprocess/tidy_libri360_kaldi_data.py $1 $2
# rm -rf libri-360

