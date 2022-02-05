from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import beta_bernoulli, dynconv


class Aligner(nn.Module):
    """Dynamic Convolution attention.
    """
    def __init__(self,
                 inputs: int,
                 channels: int,
                 loc: int,
                 kernels: int,
                 priorlen: int,
                 alpha: float,
                 beta: float):
        """Initializer.
        Args:
            inputs: I + H, size of the input tensors.
            channels: C, size of the internal hidden states.
            loc: A, size of the hiddens states for attention computation.
            kernels: size of the convolutional kernels.
            priorlen: size of the prior distribution lengths.
            alpha, beta: beta-binomial parameters.
        """
        super().__init__()
        self.loc, self.kernels = loc, kernels
        self.attn = nn.GRUCell(inputs, channels)
        self.proj_loc = nn.Sequential(
            nn.Conv1d(1, loc, kernels, padding=kernels // 2, bias=False),
            nn.Conv1d(loc, channels // 2, 1, bias=False))
        self.proj_kernel = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh(),
            nn.Linear(channels, loc * kernels, bias=False))
        self.proj_dyn = nn.Conv1d(loc, channels // 2, 1)
        self.register_buffer(
            'prior',
            # flip for pushing forward
            torch.flip(beta_bernoulli(priorlen, alpha, beta)[None, None], dims=(-1,)))
        self.aggregator = nn.Conv1d(channels // 2, 1, 1, bias=False)

    def state_init(self, encodings: torch.Tensor, mask: torch.Tensor) -> \
            Dict[str, torch.Tensor]:
        """Initialize states.
        Args:
            encodings: [torch.float32; [B, S, I]], text encodings.
            mask: [torch.float32; [B, S]], text mask.
        Returns:
            initial states.
        """
        with torch.no_grad():
            # B, S, _
            bsize, seqlen, _ = encodings.shape
            # [B, C]
            state = torch.zeros(bsize, self.attn.hidden_size, device=encodings.device)
            # [B, S]
            alpha = torch.zeros(bsize, seqlen, device=encodings.device)
            alpha[:, 0] = 1.
        return {'enc': encodings, 'mask': mask, 'state': state, 'alpha': alpha}

    def decode(self, frame: torch.Tensor, state: Dict[str, torch.Tensor]) -> \
            Dict[str, torch.Tensor]:
        """Compute align.
        Args:
            frame: [torch.float32; [B, H]], previous frame.
            state: state tensors.
        Returns:
            state: updated states.
        """
        # [B, I]
        prev = (state['enc'] * state['alpha'][..., None]).sum(dim=1)
        # [B, C]
        query = self.attn(torch.cat([frame, prev], dim=-1), state['state'])
        # [B, A, 1, K]
        kernel = self.proj_kernel(query).reshape(-1, self.loc, 1, self.kernels)
        # [B, 1, S]
        prev_alpha = state['alpha'][:, None]
        # [B, C // 2, S]
        score = self.proj_loc(prev_alpha) + self.proj_dyn(dynconv(prev_alpha, kernel))
        # [B, 1, S]
        prior = F.conv1d(F.pad(prev_alpha, [self.prior.shape[-1] - 1, 0]), self.prior)
        # [B, 1, S]
        log_prior = torch.log(prior + 1e-5)
        # [B, S]
        energy = (self.aggregator(torch.tanh(score)) + log_prior).squeeze(dim=1)
        energy.masked_fill_(~state['mask'].to(torch.bool), -np.inf)
        # [B, S]
        alpha = torch.softmax(energy, dim=-1)
        return {**state, 'state': query, 'alpha': alpha}

    def forward(self,
                encodings: torch.Tensor,
                mask: torch.Tensor,
                gt: torch.Tensor) -> torch.Tensor:
        """Compute alignment between text encodings and ground-truth mel spectrogram.
        Args:
            encodings: [torch.float32; [B, S, I]], text encodings.
            mask: [torch.float32; [B, S]], text masks.
            gt: [torch.float32; [B, T, H]], preprocessed mel-spectrogram.
        Returns:
            alpha: [torch.float32; [B, T, S]], attention alignment.
        """
        state = self.state_init(encodings, mask)
        # T x [B, S]
        alphas = []
        # T x [B, H]
        for frame in gt.transpose(0, 1):
            state = self.decode(frame, state)
            alphas.append(state['alpha'])
        # [B, T, S]
        return torch.stack(alphas, dim=1)
