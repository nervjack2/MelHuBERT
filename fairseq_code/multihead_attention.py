# MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from .fairseq_dropout import FairseqDropout
from pytorch_code import multi_head_attention_forward
from torch import Tensor, nn

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Forward option for different pruning situation
        self.skip_embed_dim_check = False
        self.need_intermediate = False
        self.context_layer_val = None

        self.reset_parameters()

    # # Skip embedding dimension check after pruning any head 
    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _set_need_intermediate(self, state:bool=False):
        self.need_intermediate = state

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        # For weight pruning
        for name in ["q", "k", "v", "out"]:
            submodule = getattr(self, name + "_proj")
            for hook in submodule._forward_pre_hooks.values():
                hook(submodule, None)

        is_tpu = query.device.type == "xla"
        if is_tpu:
            raise NotImplementedError('Do not support tpu.')

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        
        # Change self.embed_dim everytime after you have pruned on the attention head
        if not self.skip_embed_dim_check:
            assert embed_dim == self.embed_dim

        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            assert key_bsz == bsz
            assert value is not None
            assert src_len, bsz == value.shape[:2]

        assert key is not None and value is not None
     
        out =  multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            self.dropout_module.p,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training or self.dropout_module.apply_during_inference,
            key_padding_mask,
            need_weights,
            attn_mask,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            skip_embed_dim_check = self.skip_embed_dim_check,
            need_intermediate = self.need_intermediate
        )
        # Catching gradient for data-driven head pruning
        if self.need_intermediate:
            self.context_layer_val = out[2]
            if self.training:
                self.context_layer_val.retain_grad()
            return out[:2]
        else:
            return out