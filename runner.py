"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/)
    Reference author: Shu-wen (Leo) Yang (https://github.com/leo19941227), Andy T. Liu (https://github.com/andi611)
"""
import os
import math
import glob
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import MelFeatDataset
from pretrain_expert import MelHuBERTPretrainer 

class Runner():
    def __init__(self, args, runner_config):
        self.args = args
        self.runner_config = runner_config
        self.logger = SummaryWriter(args.expdir)                                                     
        self.upstream_config = yaml.load(open(self.args.upstream_config, 'r'), Loader=yaml.FullLoader)

        self.melhubert = MelHuBERTPretrainer(
            self.args,
            self.runner_config,
            self.upstream_config,
            self.args.initial_weight,
            self.args.device
        ).to(self.args.device)
        self.save_every_x_epochs = self.runner_config['runner'].get('save_every_x_epochs')
       
    def _get_optimizer(self, model):
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), **self.runner_config['optimizer'])    

        if self.args.init_optimizer_from_initial_weight:
            all_states = torch.load(self.args.initial_weight, map_location="cpu")
            init_optimizer = all_states["Optimizer"]
            try:
                optimizer.load_state_dict(init_optimizer)
                print(f'[Runner] Load initialization optimizer weight from {self.args.initial_weight}')
            except:
                raise NotImplementedError('Could not load the initialization weight of optimizer')

        return optimizer

    def _get_dataloader(self):
        dataset = MelFeatDataset(
            self.args.frame_period,
            self.runner_config['datarc']
        )
    
        dataloader = DataLoader(
            dataset, 
            batch_size=1, # for bucketing
            shuffle=True, 
            num_workers=self.runner_config['datarc']['num_workers'],
            drop_last=False, 
            pin_memory=True, 
            collate_fn=dataset.collate_fn
        )
        return dataloader

    def train(self):
        # Set model train mode
        self.melhubert.train()
        # Prepare data
        gradient_accumulate_steps = self.runner_config['runner']['gradient_accumulate_steps']
        print('[Runner] - Accumulated batch size:', 
              self.runner_config['datarc']['train_batch_size'] * gradient_accumulate_steps)
        # Get dataloader
        dataloader = self._get_dataloader()
        # Convert between pre-training epochs and total steps
        n_epochs = self.runner_config['runner']['n_epochs']
        if n_epochs > 0: 
            total_steps = int(n_epochs * len(dataloader.dataset) / gradient_accumulate_steps)
            self.runner_config['runner']['total_steps'] = total_steps
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.runner_config['runner']['total_steps']
            n_epochs = int(total_steps * gradient_accumulate_steps / len(dataloader.dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')
    
        step_per_epoch = len(dataloader.dataset)//gradient_accumulate_steps
        
        # Set optimizer
        optimizer = self._get_optimizer(self.melhubert)
        # set progress bar
        pbar = tqdm(total=self.runner_config['runner']['total_steps'], dynamic_ncols=True, desc='overall')

        all_loss = 0
        global_step = 0
        backward_steps = 0

        while pbar.n < pbar.total:
            for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                if (global_step % int(self.save_every_x_epochs * step_per_epoch) == 0) and (backward_steps % gradient_accumulate_steps == 0):
                    num_epoch = global_step // step_per_epoch
                    self.melhubert.save_model(optimizer, global_step, num_epoch)
               
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                    loss = self.melhubert(
                        data,
                        global_step=global_step,
                        log_step=self.runner_config['runner']['log_step'],
                    )

                    if gradient_accumulate_steps > 1:
                        loss = loss / gradient_accumulate_steps
                    loss.backward()

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        tqdm.write(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # Record loss
                loss_value = loss.item()
                all_loss += loss_value
                del loss
                
                # Whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue
              
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.melhubert.model.parameters(), self.runner_config['runner']['gradient_clipping'])
                if math.isnan(grad_norm):
                    tqdm.write(f'[Runner] - Error : grad norm is NaN at global step {global_step}')
                elif not math.isnan(grad_norm):
                    optimizer.step()
                optimizer.zero_grad()

                # Logging
                if global_step % self.runner_config['runner']['log_step'] == 0 or pbar.n == pbar.total -1:
                    # Log loss
                    if global_step % self.runner_config['runner']['log_step'] == 0:
                        all_loss /= self.runner_config['runner']['log_step']
                    else:
                        all_loss /= (global_step % self.runner_config['runner']['log_step'])
        
                    self.logger.add_scalar(f'loss', all_loss, global_step=global_step)
                    all_loss = 0
                    # Log norm
                    self.logger.add_scalar(f'gradient norm', grad_norm, global_step=global_step)
                
                # Save model at the last step
                if pbar.n == pbar.total-1:
                    ckpt_name = 'last-step.ckpt'
                    self.melhubert.save_model(optimizer, global_step, num_epoch, name=ckpt_name)
                
                pbar.update(1)

        pbar.close()
