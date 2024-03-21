# MelHuBERT: A simplified HuBERT on Mel spectrogram
This is the official implementation of ASRU 2023 accepted paper. 

Paper link: https://arxiv.org/abs/2211.09944

Paper introduction video: https://www.youtube.com/watch?v=S_t2TROKu6o 

MelHuBERT, is able to achieve favorable performance on phone
recognition, speaker identification, and automatic speech
recognition against HuBERT, while saving 31.2% of the pretraining time, or equivalently 33.5% MACs per one second
speech.

## Environment 
python=3.9
```
pip install -r requirement.txt
```

## Data Preparing
First, please download dataset [here](https://drive.usercontent.google.com/download?id=1Z4WU6m5v1Aq8MpzpoYIggBcLD-SQUkr9&export=download&authuser=1), and unzip the dataset.

Then, please execute the following command to prepare log Mel feature and paired cluster labels (K-means on log Mel feature)
```
bash preprocess.sh [DATASET_DIR] [OUT_DIR]
```

Then, please adjust **datarc.sets** in ./config/config_runner_20ms.yaml and ./config/config_runner_10ms.yaml to [ OUT_DIR/libri-360-data-cluster-pair.csv ]

The mean and std of LibriSpeech 360 hours is saved at OUT_DIR/mean-std.npy
(You won't need it during pre-training, but you might need it when fine-tuning on downstream.)

## Pre-training MelHuBERT from scratch
Execute the following command to pretrain MelHuBERT from scratch with default configuration

- 20 ms frame period:
```
python3 train.py -f 20 -g ./config/config_model_20ms.yaml -c ./config/config_runner_20ms.yaml -n EXP_DIR_PATH 
```
- 10 ms frame period:

```
python3 train.py -f 10 -g ./config/config_model_10ms.yaml -c ./config/config_runner_10ms.yaml -n EXP_DIR_PATH 
```

-f: frame period \
-g: Model config \
-c: Runner config \
-n: The model checkpoints, log file, and the pre-training config you used will be saved at this directory 

## Pretrained Models 
**Warning: Note that these models are trained with 32 batch size, which is much smaller than fairseq's HuBERT Base. So they could not directly compare to HuBERT Base.**
- [MelHuBERT-20ms 360-hour stage 1](https://drive.google.com/file/d/1mSR40Vdl2gT1rlZORleKPb2gcryQHW5m/view?usp=sharing)
- [MelHuBERT-20ms 360-hour stage 2](https://drive.google.com/file/d/11wzYf8u9pXPvQyQU2Wodx79W31Ka2e0Z/view?usp=sharing)
- [MelHuBERT-10ms 360-hour stage 1](https://drive.google.com/file/d/1YZP9nBSRaQ_Z2cFaFLmLkGilEYsEHb2b/view?usp=sharing)
- [MelHuBERT-20ms 100-hour stage 1](https://drive.google.com/file/d/1ppz5w5eTGOL81QjYqwxRwFFmq-hqInD6/view?usp=sharing)
- [MelHuBERT-10ms 960-hour stage 2](https://drive.google.com/file/d/18u2u-528uDh5T7R1bp1wvWJ2ygcrNlzx/view?usp=sharing)
- [MelHuBERT-20ms 960-hour stage 2](https://drive.google.com/file/d/1Fn_C5VoH5iV3LdvBEjvfAsbMPhWFFPdd/view?usp=sharing)

## Extracting feature 
Please execute the following command to extract feature from two example waveforms
```
python3 extract_feature.py -c [CHECKPOINT] -f [FRAME_PERIOD]
```

-c: Model checkpoint path
-f: Choice from 20 or 10 (ms)

## Acknowledgement 
Our implementation of pre-training interface is based on [S3PRL toolkit](https://github.com/s3prl/s3prl)
