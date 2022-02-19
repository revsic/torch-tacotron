from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Aligner(nn.Module):
    """Forward attention.
    """
    def __init__(self,
                 inputs: int,
                 queries: int,
                 outputs: int,
                 channels: int,
                 loc: int,
                 kernels: int):
        """Initializer.
        Args:
            inputs: I, size of the input tensors.
            queries: H, size of the query tensor.
            outputs: M, size of the output tensor from spectrogram decoder.
            channels: C, size of the internal hidden states.
            loc: A, size of the hiddens states for attention computation.
            kernels: size of the convolutional kernels.
        """
        super().__init__()
        self.loc, self.kernels = loc, kernels
        self.attn = nn.GRUCell(inputs + queries, channels)
        self.proj_query = nn.Linear(channels, channels // 2, bias=False)
        self.proj_loc = nn.Sequential(
            nn.Conv1d(1, loc, kernels, padding=kernels // 2, bias=False),
            nn.Conv1d(loc, channels // 2, 1, bias=False))
        self.proj_key = nn.Linear(inputs, channels // 2)
        self.aggregator = nn.Conv1d(channels // 2, 1, 1, bias=False)

        self.trans = nn.Sequential(
            nn.Linear(inputs + channels + outputs, channels),
            nn.Tanh(),
            nn.Linear(channels, 1, bias=False),
            nn.Sigmoid())

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
            align = torch.full((bsize, seqlen), seqlen ** -1, device=encodings.device)
            # [B, S]
            alpha = torch.zeros(bsize, seqlen, device=encodings.device)
            alpha[:, 0] = 1.
            # [B, 1]
            prob = torch.full((bsize, 1), 0.5, device=encodings.device)
            # [B, S], inverse the binary mask for method `masked_fill_`
            mask = ~mask.to(torch.bool)
        # [B, C // 2, S]
        key = self.proj_key(encodings).transpose(1, 2)
        return {
            'enc': encodings, 'key': key, 'mask': mask, 'state': state,
            'align': align, 'trans': prob, 'alpha': alpha}

    def decode(self,
               query: torch.Tensor,
               frame: torch.Tensor,
               state: Dict[str, torch.Tensor]) -> \
            Dict[str, torch.Tensor]:
        """Compute align.
        Args:
            query: [torch.float32; [B, H]], query frame.
            frame: [torch.float32; [B, M]], previous output frame.
            state: state tensors.
        Returns:
            state: updated states.
        """
        # [B, I]
        prev = (state['enc'] * state['alpha'][..., None]).sum(dim=1)
        # [B, C]
        query = self.attn(torch.cat([query, prev], dim=-1), state['state'])
        # [B, C // 2, S]
        score = self.proj_loc(state['align'][:, None]) \
            + self.proj_query(query)[..., None] \
            + state['key']  # biased by proj_key
        # [B, S]
        energy = self.aggregator(torch.tanh(score)).squeeze(dim=1)
        energy.masked_fill_(state['mask'], -np.inf)
        # [B, S]
        align = torch.softmax(energy, dim=-1)
        # [B, S]
        alpha = (
            (1 - state['trans']) * state['alpha'] +  # keep + transition + eps
            state['trans'] * F.pad(state['alpha'], [1, -1]) + 1e-7) * align
        # normalize
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        # [B, I]
        attend = (state['enc'] * alpha[..., None]).sum(dim=1)
        # [B, 1]
        trans = self.trans(torch.cat([attend, frame, query], dim=-1))
        return {**state, 'state': query, 'align': align, 'trans': trans, 'alpha': alpha}

    def forward(self,
                encodings: torch.Tensor,
                queries: torch.Tensor,
                outputs: torch.Tensor,
                mask: torch.Tensor,) -> torch.Tensor:
        """Compute alignment between text encodings and ground-truth mel spectrogram.
        Args:
            encodings: [torch.float32; [B, S, I]], text encodings.
            queries: [torch.float32; [B, T, H]], query input, preprocessed mel-spectrogram.
            outputs: [torch.float32; [B, T, M]], raw mel-spectrogram.
            mask: [torch.float32; [B, S]], text masks.
        Returns:
            alpha: [torch.float32; [B, T, S]], attention alignment.
        """
        state = self.state_init(encodings, mask)
        # T x [B, S]
        alphas = []
        # T x ([B, H], [B, M]
        for query, frame in zip(queries.transpose(0, 1), outputs.transpose(0, 1)):
            state = self.decode(query, frame, state)
            alphas.append(state['alpha'])
        # [B, T, S]
        return torch.stack(alphas, dim=1)
