import sys 
import os 
from tqdm import tqdm
import torchaudio
import torch
import numpy as np 

def read_audio(path, ref_len=None):
    wav, sr = torchaudio.load(path)
    wav = wav*(2**15)
    assert sr == 16000, sr
        
    if ref_len is not None and abs(ref_len - len(wav[0])) > 160:
        logging.warning(f"ref {ref_len} != read {len(wav[0])} ({path})")

    return wav


def get_feats(path, mel_dim=40, ref_len=None):
    x = read_audio(path, ref_len)
    with torch.no_grad():
        y = torchaudio.compliance.kaldi.fbank(
                x,
                num_mel_bins=mel_dim,
                sample_frequency=16000,
                window_type='hamming',
                frame_length=25,
                frame_shift=10,
        )
        y = y.contiguous()
    y = np.array(y)

    return y

data_tsv_pth, mean_var_save_path, mel_dim = sys.argv[1], sys.argv[2], sys.argv[3]
mel_dim = int(mel_dim)

file_pth = []
with open(data_tsv_pth, 'r') as fp:
    root_path = fp.readline().strip()
    for x in fp:
        file_pth.append(x.strip().split(' ')[0])
file_pth = [os.path.join(root_path, x) for x in file_pth]

sum_ = np.zeros((1,mel_dim))
sum_square = np.zeros((1,mel_dim))
total_count = 0 

for pth in tqdm(file_pth):
    feat = get_feats(pth, mel_dim=mel_dim)
    sum_ += np.sum(feat, axis=0)
    sum_square += np.sum(feat**2, axis=0)
    total_count += len(feat)
        
mean = sum_ / total_count
std = ((sum_square/total_count)-(mean**2))**(1/2)
mean_std = np.concatenate((mean, std), axis=0)
np.save(mean_var_save_path, mean_std)
            
