import sys
import os  
from tqdm import tqdm
import kaldiark
import numpy as np 

def read_scp_file(scp_path, data_dir):
    with open(scp_path, 'r') as fp:
        data = {}
        for x in fp:
            key, path = x.split(' ')
            name, bits = path.split('/')[-1].split(':')
            data[key] = (os.path.join(data_dir, name), int(bits))
    return data 

def read_mean_var(path):
    with open(path, 'r') as fp:
        sum_ = np.fromstring(fp.readline().strip()[1:-1], dtype=float, sep=',')
        sum_sqr =  np.fromstring(fp.readline().strip()[1:-1], dtype=float, sep=',')
        n_frame = int(fp.readline().strip())
        mean = sum_/n_frame
        std = np.power((sum_sqr/n_frame)-(np.power(mean,2)), 1/2)
        return mean, std

def main(
    data_dir,
    out_dir,
):
    key_data_path = os.path.join(data_dir, 'train-clean-360.scp')
    key_label_path = os.path.join(data_dir, 'train-clean-360-k512-e10.bas.scp')
    mean_var_path = os.path.join(data_dir, 'train-clean-360.mean-var')
    data_save_dir = os.path.join(out_dir, 'feature')
    label_save_dir = os.path.join(out_dir, 'cluster')
    mean_std_npy_save_path = os.path.join(out_dir, 'mean-std.npy')
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'libri-360-data-cluster-pair.csv')
    out_fp = open(out_csv, 'w')
    mean, std = read_mean_var(mean_var_path)
    mean_std = np.concatenate((mean.reshape(1,-1), std.reshape(1,-1)), axis=0)
    np.save(mean_std_npy_save_path, mean_std)
    data_dict = read_scp_file(key_data_path, data_dir)
    label_dict = read_scp_file(key_label_path, data_dir)
    recorder = {}
    
    for key, values in tqdm(data_dict.items()):
        data_path, bits = values[0], values[1]
        with open(data_path, 'rb') as fp:
            fp.seek(bits)
            feat = kaldiark.parse_feat_matrix(fp)
            feat = (feat-mean)/std
            length = feat.shape[0]
            save_path = os.path.join(data_save_dir, key+'.npy')
            recorder[key] = [save_path, length]
            np.save(save_path, feat)
    
    for key, values in tqdm(label_dict.items()):
        data_path, bits = values[0], values[1]
        with open(data_path, 'r') as fp:
            fp.seek(bits)
            label = np.array(list(map(int, fp.readline().strip().split(' '))))
            assert not ((label >= 512).any() or (label < 0).any())
            length = label.shape[0]
            save_path = os.path.join(label_save_dir, key+'.npy')
            recorder[key].append(save_path)
            assert recorder[key][1] == length
            np.save(save_path, label)

    out_fp.write('file_path,label_path,length\n')
    for data_path, length, label_path in recorder.values():
        out_fp.write(f'{data_path},{label_path},{length}\n')

if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    main(data_dir, out_dir)
