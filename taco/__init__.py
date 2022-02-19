from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbhg import Cbhg
from .decoder import Decoder
from .misc import Prenet, Reduction
from .config import Config


class Tacotron(nn.Module):
    """Tacotron for multispeakers
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configurations.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocabs, config.embeddings, padding_idx=0)
        self.prenet = Prenet(
            config.embeddings,
            config.enc_prenet + [config.channels // 2],
            config.enc_dropout)

        self.cbhg = Cbhg(
            config.channels // 2,
            config.cbhg_banks,
            config.cbhg_pool,
            config.cbhg_kernels,
            config.cbhg_highways)

        # padding with silence
        self.reduction = Reduction(config.reduction, value=np.log(1e-5))

        self.teacher_force = config.teacher_force

        self.decoder = Decoder(
            config.channels,
            config.channels,
            config.dec_prenet,
            config.fwd_loc,
            config.fwd_kernels,
            config.dec_dropout,
            config.dec_layers,
            config.reduction * config.mel,
            config.dec_max_factor)

    def forward(self,
                text: torch.Tensor,
                textlen: torch.Tensor,
                mel: Optional[torch.Tensor] = None,
                mellen: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode text tokens.
        Args;
            text: [torch.long; [B, S]], text symbol sequences.
            textlen: [torch.long; [B]], sequence lengths.
            mel: [torch.float32; [B, T, M]], mel-spectrogram, if provided.
            mellen: [torch.long; [B]], spectrogram lengths, if provided.
        Returns:
            mel: [torch.float32; [B, T, M]], predicted spectrogram.
            mellen: [torch.long; [B]], spectrogram lengths.
            auxiliary: auxiliary informations.
                align: [torch.float32; [B, T // F, S]], attention alignments.
                unmasked: [torch.float32; [B, T, M]], unmasked predicted spectrogram.
        """
        ## 1. Text encoding
        # pad for eos check, default zero for pad
        text = F.pad(text, [0, 1])
        # S
        seqlen = text.size(1)
        # [B, S]
        text_mask = (
            torch.arange(seqlen, device=text.device)[None]
            < textlen[:, None]).to(torch.float32)
        # [B, S, E]
        embed = self.embedding(text)
        # [B, S, C // 2], masking for initial convolution of CBHG.
        preproc = self.prenet(embed) * text_mask[..., None]
        # [B, S, C]
        encodings = self.cbhg(preproc)

        ## 3. Decoding
        if mel is not None:
            # [B, T // F, F x M]
            mel, remains = self.reduction(mel)
        else:
            remains = None
        # [B, T // F, F x M]
        mel, aux = self.decoder(encodings, text_mask, gt=mel) \
            if self.teacher_force is None \
            else self.decoder.greedyfwd(encodings, text_mask, gt=mel, prob=self.teacher_force)

        ## 4. Unfold
        if mellen is None:
            mellen = aux['mellen'].to(torch.long) * self.reduction.factor
            del aux['mellen']
        # [B, T, M]
        mel = self.reduction.unfold(mel, remains)
        # [B, T]
        mel_mask = (
            torch.arange(mel.size(1), device=mel.device)[None]
            < mellen[:, None]).to(torch.float32)
        # mask with silence
        masked_mel = mel.masked_fill(~mel_mask[..., None].to(torch.bool), np.log(1e-5))
        # [B, T, M]
        return masked_mel, mellen, {**aux, 'unmasked': mel}

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict()}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    def load(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])
