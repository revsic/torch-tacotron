from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Aligner(nn.Module):
    """Forward attention.
    """
    def __init__(self, inputs: int, queries: int, channels: int, loc: int, kernels: int):
        """Initializer.
        Args:
            inputs: I, size of the input tensors.
            queries: H, size of the input query.
            channels: C, size of the internal hidden states.
            loc: L, size of the hiddens states for attention computation.
            kernels: size of the convolutional kernels.
        """
        super().__init__()
        self.attn = nn.GRUCell(inputs + queries, channels)
        self.proj_query = nn.Linear(channels, loc, bias=False)
        self.proj_key = nn.Linear(inputs, loc, bias=False)
        self.proj_loc = nn.Conv1d(1, loc, kernels, padding=kernels // 2)
        self.aggregator = nn.Linear(loc, 1)

        self.trans = nn.Sequential(
            nn.Linear(inputs + queries + channels, channels),
            nn.Tanh(),
            nn.Linear(channels, 2),
            nn.Softmax(dim=-1))

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
            # [B, S]
            align = alpha.clone()
        return {'enc': encodings, 'mask': mask, 'state': state, 'alpha': alpha, 'align': align}

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
        # [B, S, A]
        score = self.proj_query(query)[:, None] + self.proj_key(state['enc']) \
            + self.proj_loc(state['align'][:, None]).transpose(1, 2)
        # [B, S]
        energy = self.aggregator(torch.tanh(score)).squeeze(dim=-1)
        energy.masked_fill_(~state['mask'].to(torch.bool), -np.inf)
        # [B, S]
        align = torch.softmax(energy, dim=-1)
        # [B, 1]
        stop, next_ = self.trans(torch.cat([frame, prev, query], dim=-1)).chunk(2, dim=-1)
        # [B, S]
        alpha = (stop * state['alpha'] + next_ * F.pad(state['alpha'], [1, -1]) + 1e-5) * align
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        return {**state, 'state': query, 'alpha': alpha, 'align': align}

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
