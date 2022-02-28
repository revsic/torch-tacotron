from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from taco import Tacotron


class TrainingWrapper:
    """Tacotron training wrapper.
    """
    def __init__(self, model: Tacotron, device: torch.device):
        """Initializer.
        Args:
            model: tacotron model.
            device: torch device.
        """
        self.model = model
        self.device = device

    def wrap(self, bunch: List[np.ndarray]) -> List[torch.Tensor]:
        """Wrap the array to torch tensor.
        Args:
            bunch: input tensors.
        Returns:
            wrapped.
        """
        return [torch.tensor(array, device=self.device) for array in bunch]

    def compute_loss(self, bunch: List[np.ndarray]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute unconditional Tacotron loss.
        Args:
            bunch: input tensors.
                text: [np.long; [B, S]], input text tokens.
                textlen: [np.long; [B]], lengths of each texts.
                mel: [np.float32; [B, T, M]], mel-spectrograms.
                mellen: [np.long; [B]], length of each mel-spectrograms.
        Returns:
            loss tensor and details.
        """
        ## wrapping
        text, mel, textlen, mellen = self.wrap(bunch)
        # [B, T]
        mel_mask = (
            torch.arange(mel.shape[1], device=mel.device)[None]
            < mellen[:, None].to(torch.float32))
        # mask with silence
        # => additional padding for spectrogram reduction should use silence pad value
        mel.masked_fill_(~mel_mask[..., None].to(torch.bool), np.log(1e-5))

        ## outputs
        masked_mel, _, aux = self.model(text, textlen, mel, mellen)
        # mel spectrogram loss
        loss = F.l1_loss(mel, aux['unmasked'])
        aux = {'mel': masked_mel, 'align': aux['align']}
        return loss, aux
