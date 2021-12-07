from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .nat import NonAttentiveTacotron


class TacoSpawn(nn.Module):
    """TacoSpawn variations of Non-attentive Tacotron.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.nat = NonAttentiveTacotron(config)
        # unconditional prior space, mean and log stddev
        self.num_modals = config.modal
        self.priorbuffer = nn.Parameter(
            torch.randn(config.modal, config.spkembed * 2 + 1),
            requires_grad=True)
        # speaker embedding
        self.spkbuffer = nn.Parameter(
            torch.randn(config.speakers, config.spkembed * 2 + 1),
            requires_grad=True)

    def forward(self,
                text: torch.Tensor,
                textlen: torch.Tensor,
                mel: Optional[torch.Tensor] = None,
                mellen: Optional[torch.Tensor] = None,
                spkid: Optional[torch.Tensor] = None,
                modalid: Optional[torch.Tensor] = None,
                sample: bool = True) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode text tokens.
        Args;
            text: [torch.long; [B, S]], text symbol sequences.
            textlen: [torch.long; [B]], sequence lengths.
            mel: [torch.float32; [B, T, M]], mel-spectrogram, if provided.
            mellen: [torch.long; [B]], spectrogram lengths, if provided.
            spkid: [torch.long; [B]], speaker id, if provided.
            modalid: [torch.long; [B]], modal id, if provided,
                if both spkid and modalid provided, return sample based on spkid.
            sample: whether sample from distribution or use mean.
        Returns:
            mel: [torch.float32; [B, T, M]], predicted spectrogram.
            mellen: [torch.long; [B]], spectrogram lengths.
            auxiliary: auxiliary informations.
                spkembed: [torch.float32; [B, E]], sampled speaker embedding.
                align: [torch.float32; [B, T // F, S]], attention alignments.
                durations: [torch.float32; [B, S]], durations.
                factor: [torch.float32; [B]], size ratio between ground-truth and predicted lengths.
        """
        # sample speaker embedding
        if spkid is None and modalid is None:
            # sample random modal
            modalid = torch.randint(
                self.num_modals, (text.shape[0],), device=text.device)

        # [B], [K, E + 1]
        ids, buffer = (spkid, self.spkbuffer) \
            if spkid is not None else (modalid, self.priorbuffer)
        # [K], [K, E], [K, E]
        _, mean, std = self.parametrize(buffer)
        # [B, E]
        spkembed = mean[ids] + torch.randn_like(std[ids]) * (
            std[ids] if sample else 0.)
        # [B, T, M], [B], _
        mel, mellen, aux = self.nat(text, textlen, spkembed, mel, mellen)
        return mel, mellen, {'spkembed': spkembed, **aux}

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

    def parametrize(self, buffer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Speaker prior.
        Args:
            buffer: [torch.float32; [K, E x 2 + 1]], distribution weights.
        Returns:
            weight: [torch.float32; [K]], weights of each modals.
            mean: [torch.float32; [K, E]], mean vectors.
            std: [torch.float32; [K, E]], standard deviations.
        """
        # [K]
        weight = torch.softmax(buffer[:, 0], dim=0)
        # [K, E], [K, E]
        mean, logstd = buffer[:, 1:].chunk(2, dim=-1)
        # [K, E]
        std = F.softplus(logstd)
        # [K], [K, E], [K, E]
        return weight, mean, std
