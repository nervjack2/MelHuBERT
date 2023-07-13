"""
This is a simple example of how to extract feature.
Please use -m option to specify the mode.
"""

import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np 
from torch.nn.utils.rnn import pad_sequence
from model import MelHuBERTConfig, MelHuBERTModel

def load_mean_std(mean_std_npy_path):
    mean_std = np.load(mean_std_npy_path)
    mean = torch.Tensor(mean_std[0].reshape(-1))
    std = torch.Tensor(mean_std[1].reshape(-1))
    return mean, std  

def extract_fbank(waveform_path, mean, std, fp=20):
    waveform, sr = torchaudio.load(waveform_path)
    waveform = waveform*(2**15)
    y = torchaudio.compliance.kaldi.fbank(
                        waveform,
                        num_mel_bins=40,
                        sample_frequency=16000,
                        window_type='hamming',
                        frame_length=25,
                        frame_shift=10)
    # Normalize by the mean and std of Librispeech
    mean = mean.to(y.device, dtype=torch.float32)
    std = std.to(y.device, dtype=torch.float32)
    y = (y-mean)/std
    # Downsampling by twice 
    if fp == 20:
        odd_y = y[::2,:]
        even_y = y[1::2,:]
        if odd_y.shape[0] != even_y.shape[0]:
            even_y = torch.cat((even_y, torch.zeros(1,even_y.shape[1]).to(y.device)), dim=0)
        y = torch.cat((odd_y, even_y), dim=1)
    return y

def prepare_data(wav_path, fp=20):
    # Load the mean and std of LibriSpeech 360 hours 
    mean_std_npy_path = './example/libri-360-mean-std.npy'
    mean, std = load_mean_std(mean_std_npy_path)
    # Extract fbank feature for model's input
    mel_input = [extract_fbank(p, mean, std, fp) for p in wav_path]
    mel_len = [len(mel) for mel in mel_input]
    mel_input = pad_sequence(mel_input, batch_first=True) # (B x S x D)
    # Prepare padding mask
    pad_mask = torch.ones(mel_input.shape[:-1])  # (B x S)
    # Zero vectors for padding dimension
    for idx in range(mel_input.shape[0]):
        pad_mask[idx, mel_len[idx]:] = 0

    return mel_input, mel_len, pad_mask


def main(args):
    # Preparing example input
    wav_path = [
        './example/100-121669-0000.flac',
        './example/1001-134707-0000.flac'
    ]
    print(f'[Extractor] - Extracting feature from these files: {wav_path}')
    mel_input, mel_len, pad_mask = prepare_data(wav_path, args.fp)
    # Put data on device 
    mel_input = mel_input.to(
        device=args.device, dtype=torch.float32
    )  
    pad_mask = torch.FloatTensor(pad_mask).to( 
        device=args.device, dtype=torch.float32
    )  
    
    # Load upstream model 
    all_states = torch.load(args.checkpoint, map_location="cpu")
    upstream_config = all_states["Upstream_Config"]["melhubert"]  
    upstream_config = MelHuBERTConfig(upstream_config)
    upstream_model = MelHuBERTModel(upstream_config).to(args.device)
    state_dict = all_states["model"]
    upstream_model.load_state_dict(state_dict)
    upstream_model.eval() 
    total_params = sum(p.numel() for p in upstream_model.parameters())
    print(f'[Extractor] - Successfully load model with {total_params} parameters')

    with torch.no_grad():
        out = upstream_model(mel_input, pad_mask, get_hidden=True, no_pred=True)

    last_layer_feat, hidden_states = out[0], out[5]
    print(f'[Extractor] - Feature with shape of {last_layer_feat.shape} is extracted')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Path to model checkpoint')
    parser.add_argument('-f', '--fp', type=int, help='frame period', default=20)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    args = parser.parse_args()
    main(args)
