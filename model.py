"""
    Model config and model structure of MelHuBERT. 
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)
    Reference author: Heng-Jui Chang (https://github.com/vectominist)
"""
import numpy as np
import torch
from torch import nn
from module import TransformerEncoder
from fairseq_code import compute_mask_indices

class MelHuBERTConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Input feature dimemsion 
        self.feat_emb_dim = int(config.get("feat_emb_dim", 40))
       
        # Positional embedding type
        self.pos_emb_type = str(config.get("pos_emb_type", "conv"))
        self.pos_conv_depth = int(config.get("pos_conv_depth", 1))
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))
        # Output dimension 
        self.num_cluster = int(config.get("num_cluster", 512))
        self.final_dim = int(config.get("final_dim", 40))
        # Criterion (This two parameters would not be used in distillation mode)
        self.pred_masked_weight = float(config.get("pred_masked_weight", 1.0))
        self.pred_nomask_weight = float(config.get("pred_nomask_weight", 0.0))
        # Masking 
        self.mask_prob = float(config.get("mask_prob", 0.8))
        self.mask_length = int(config.get("mask_length", 10))
        self.mask_selection = str(config.get("mask_selection", 'static'))
        self.mask_other = float(config.get("mask_other", 0.0))
        self.no_mask_overlap = bool(config.get("no_mask_overlap", False))
        self.mask_min_space = int(config.get("mask_min_space", 1))

        self.skip_masked = bool(config.get("skip_masked", False))
        self.skip_nomask = bool(config.get("skip_nomask", True))

        self.learnable_mask_emb = bool(config.get("learnable_mask_emb", False))
        self.mask_before_proj = bool(config.get("mask_before_proj", True))
        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

class MelHuBERTModel(nn.Module):

    def __init__(self, model_config: MelHuBERTConfig, multitask=False):
        super().__init__()

        self.model_config = model_config

        self.n_encoder_layers = model_config.encoder_layers
        print(
            f"[MelHuBERTModel] - Encoder layer = {self.n_encoder_layers}"
        )

        self.pre_extract_proj = (
            nn.Linear(model_config.feat_emb_dim,  model_config.encoder_embed_dim)
            if model_config.feat_emb_dim != model_config.encoder_embed_dim
            else None
        )

        if model_config.encoder_layers > 0:
            self.encoder = TransformerEncoder(model_config)
        else:
            self.encoder = nn.GELU()
        
        if self.model_config.learnable_mask_emb:
            if not self.model_config.mask_before_proj: 
                self.mask_emb = nn.Parameter(
                    torch.FloatTensor(model_config.encoder_embed_dim).uniform_().to('cuda')
                )
            else:
                self.mask_emb = nn.Parameter(
                    torch.FloatTensor(model_config.feat_emb_dim).uniform_().to('cuda')
                )
        else:
            if not self.model_config.mask_before_proj:
                self.mask_emb = 0
            else:
                self.mask_emb = 0

        self.final_proj = nn.Linear(model_config.encoder_embed_dim, model_config.num_cluster)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.model_config.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.model_config.mask_prob,
                self.model_config.mask_length,
                self.model_config.mask_selection,
                self.model_config.mask_other,
                min_masks=2,
                no_overlap=self.model_config.no_mask_overlap,
                min_space=self.model_config.mask_min_space,
                require_same_masks=False
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices =  None

        return x, mask_indices

    def forward(
        self, 
        feat, 
        pad_mask, 
        cluster_label=None, 
        no_pred=False,
        mask=False, 
        get_hidden=False, 
    ):
        """
        Forward function
        Input:
            feat (FloatTensor): B x T_wave x D
            pad_mask (BoolTensor): B x T_wave
        """
        # Masking before projection 
        if mask and self.model_config.mask_before_proj:
            input_feat, mask_indices = self.apply_mask(feat, ~pad_mask.bool())
        else:
            input_feat = feat
            mask_indices = torch.full(pad_mask.shape, False)

        pre_feat = input_feat
        if self.pre_extract_proj != None:
            pre_feat = self.pre_extract_proj(input_feat)
        
        # Masking after projection 
        if mask and not self.model_config.mask_before_proj:
            x, mask_indices = self.apply_mask(pre_feat, ~pad_mask.bool(), teacher_mask_indices)
        else:
            x = pre_feat
            mask_indices = mask_indices
        
        layer_hiddens = []
        hidden, layer_hiddens = self.encoder(
            x, ~pad_mask.bool(), get_hidden=get_hidden
        )
        
        if no_pred:
            return hidden, None, None, None, None, layer_hiddens, pre_feat

        assert cluster_label != None

        if not self.model_config.skip_masked:
            masked_indices = torch.logical_and(pad_mask.bool(), mask_indices)
            logit_m = self.final_proj(hidden[masked_indices])  # (num_masked, dim) -> (num_masked, num_cluster)
            label_m = cluster_label[masked_indices]
        else:
            logit_m = None
            label_m = None

        if not self.model_config.skip_nomask:
            nomask_indices = torch.logical_and(pad_mask.bool(), ~mask_indices) 
            logit_u = self.final_proj(hidden[nomask_indices])  # (num_unmask, dim) -> (num_unmask, num_cluster)
            label_u = cluster_label[nomask_indices]
        else:
            logit_u = None
            label_u = None
        
        return hidden, logit_m, logit_u, label_m, label_u, layer_hiddens, pre_feat, mask_indices
