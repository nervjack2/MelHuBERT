"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl)
    Reference author: Shu-wen (Leo) Yang (https://github.com/leo19941227), Andy T. Liu (https://github.com/andi611)
"""
import yaml
import os 
import torch
import torch.nn as nn
from model import MelHuBERTModel, MelHuBERTConfig

class MelHuBERTPretrainer(nn.Module):
    def __init__(self, args, runner_config, upstream_config, initial_weight=None, device='cuda'):
        super(MelHuBERTPretrainer, self).__init__()

        self.args = args
        self.runner_config = runner_config
        self.upstream_config = upstream_config
        self.initial_weight = initial_weight
        self.device = device

        # Initialize the model 
        self._init_model()
        # Define pre-training loss
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        print('[Pretrainer] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):
        print('[Pretrainer] - Initializing model...')
        self.model_config = MelHuBERTConfig(self.upstream_config['melhubert'])
        self.model = MelHuBERTModel(self.model_config)

        # Do initialization from a checkpoint if needed
        if self.initial_weight:
            all_states = torch.load(self.initial_weight, map_location="cpu")
            try:             
                self.model.load_state_dict(all_states["model"])
                print(f'[Pretrainer] Load initilization model weight from {self.initial_weight}')
            except:
                raise NotImplementedError('Could not load the initilization weight')

    def save_model(self, optimizer, global_step, num_epoch=-1, name=None):
        if global_step == 0:
            return 
        all_states = {
            'Optimizer': optimizer.state_dict(),
            'Step': global_step,
            'Args': self.args,
            'Runner': self.runner_config,
            'model': self.model.state_dict(),
            'Upstream_Config': self.upstream_config
        }
        if not name:
            name = f'checkpoint-epoch-{num_epoch}.ckpt'
        save_path = os.path.join(self.args.expdir, name)
        tqdm.write(f'[MelHuBERT] - Save the checkpoint to: {save_path}')
        torch.save(all_states, save_path)

    def forward(self, data, global_step=0, log_step=1000):
        """
        Args:
            data: [audio feature, cluster id, padding mask, audio length]
        Return:
            loss        
        """
        audio_feat, label, pad_mask, audio_len = data[0], data[1], data[2], data[3]

        audio_feat = audio_feat.to(self.device)
        label = label.to(self.device)
        pad_mask = pad_mask.to(self.device)
  
        _, logit_m, logit_u, label_m, label_u, _, _, _ = self.model(audio_feat, pad_mask, label, mask=True)
        loss = 0.0 
        if logit_m != None and label_m != None and self.model_config.pred_masked_weight > 0: 
            loss += self.model_config.pred_masked_weight * self.loss(logit_m, label_m)   
        if logit_u != None and label_u != None and self.model_config.pred_nomask_weight > 0: 
            loss += self.model_config.pred_nomask_weight * self.loss(logit_u, label_u)
        
        return loss
        
