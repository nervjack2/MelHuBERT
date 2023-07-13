# MelHuBERT: A simplified HuBERT on Mel spectrogram
This is the official implementation of https://arxiv.org/abs/2211.09944

## Data Preparing
First, please execute the following command to prepare LibriSpeech 360 horus and paired cluster labels (K-means on log Mel feature)
```
bash preprocess.sh [DATA_DIR]
```

Then, please adjust **datarc.sets** in config_runner.yaml to [ DATA_DIR/libri-360-data-cluster-pair.csv ]

The mean and std of LibriSpeech 360 hours is saved at DATA_DIR/mean-std.npy

## Pre-training MelHuBERT from scratch
Execute the following command to pretrain MelHuBERT from scratch with default configuration
```
python3 train.py -m melhubert -g ./config/config_model.yaml -c ./config/config_runner.yaml -n EXP_DIR_PATH 
```
-g: Model config \
-c: Runner config \
-n: The model checkpoints, log file, and the pre-training config you used will be saved at this directory 

## Pretrained Models 
- [MelHuBERT-20ms 360-hour stage 1](https://drive.google.com/file/d/1mSR40Vdl2gT1rlZORleKPb2gcryQHW5m/view?usp=sharing)
- [MelHuBERT-20ms 360-hour stage 2](https://drive.google.com/file/d/11wzYf8u9pXPvQyQU2Wodx79W31Ka2e0Z/view?usp=sharing)
- [MelHuBERT-10ms 360-hour stage 1](https://drive.google.com/file/d/1ppz5w5eTGOL81QjYqwxRwFFmq-hqInD6/view?usp=sharing)
- [MelHuBERT-20ms 100-hour stage 1](https://drive.google.com/file/d/1YZP9nBSRaQ_Z2cFaFLmLkGilEYsEHb2b/view?usp=sharing)
## Extracting feature 
Please execute the following command to extract feature from two example waveforms
```
python3 extract_feature.py -c [CHECKPOINT] -f [FRAME_PERIOD]
```

-c: Model checkpoint path
-f: Choice from 20 or 10 (ms)

## Acknowledgement 
Our implementation of pre-training interface is based on [S3PRL toolkit](https://github.com/s3prl/s3prl)
