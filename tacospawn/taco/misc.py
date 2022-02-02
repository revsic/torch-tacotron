from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def beta_bernoulli(length: int, alpha: float, beta: float) -> torch.Tensor:
    """Compute beta bernoulli distribution.
    Args:
        length: K, total length.
        alpha, beta: beta-binomial parameters.
    Returns:
        [torch.float32; [K]], beta bernoulli values.
    """
    def logbeta(a, b): return a.lgamma() + b.lgamma() - (a + b).lgamma()
    # log(n! / (k! x (n - k)!))
    # = logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1)
    # = logGamma(n + 2) - log(n + 1) - logGamma(k + 1) - logGamma(n - k + 1)
    # = -logbeta(k + 1, n - k + 1) - log(n + 1) 
    def logcomb(n, k): return -logbeta(k + 1, n - k + 1) - np.log(n + 1)
    # since sequence consists of [0, n]
    n = torch.tensor(length - 1, dtype=torch.float32)
    alpha, beta = torch.tensor(alpha), torch.tensor(beta)
    # [K]
    k = torch.arange(length, dtype=torch.float32)
    return torch.exp(
        logcomb(n, k) + logbeta(k + alpha, n - k + beta) - logbeta(alpha, beta))


def dynconv(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Dynamic convolution.
    Args:
        inputs: [torch.float32; [B, I, S]], input tensor.
        weight: [torch.float32; [B, O, I, K]], weight tensor, assume K is odd.
    Returns:
        [torch.Tensor, [B, O, S]], convolved.
    """
    # B, I, S
    bsize, in_channels, seqlen = inputs.shape
    # _, O, _, K
    _, out_channels, _, kernels = weights.shape
    # [1, B x O, S]
    out = F.conv1d(
        # [1, B x I, S]
        inputs.view(1, bsize * in_channels, seqlen),
        # [B x O, I, K]
        weights.view(bsize * out_channels, in_channels, kernels),
        # set group size to batch size for
        padding=kernels // 2, groups=bsize)
    # [B, O, S]
    return out.view(bsize, out_channels, seqlen)


class Prenet(nn.Sequential):
    """Bottleneck with dropout.
    """
    def __init__(self, channels: int, hiddens: List[int], dropout: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            hiddens: size of the hidden channels.
            dropout: dropout rate.
        """
        super().__init__(*[
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout))
            for in_channels, out_channels in zip([channels] + hiddens, hiddens)])


class Reduction(nn.Module):
    """Fold the inupts, applying reduction factor.
    """
    def __init__(self, factor: int, value: float = 0.):
        """Initializer.
        Args:
            factor: reduction factor.
            value: padding value.
        """
        super().__init__()
        self.factor = factor
        self.value = value

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[int]]:
        """Fold the inputs, apply reduction factor.
        Args:
            input: [torch.float32; [B, T, C]], input tensor.
        Returns:
            [torch.float32; [B, T // F, F x C]] folded tensor and remains.
        """
        # B, T, C
        bsize, timesteps, channels = inputs.shape
        if timesteps % self.factor > 0:
            remains = self.factor - timesteps % self.factor
            # [B, T + R, C]
            inputs = F.pad(inputs, [0, 0, 0, remains], value=self.value)
        else:
            # no remains
            remains = None
        # [B, T // F, F x C]
        return inputs.reshape(bsize, -1, self.factor * channels), remains

    def unfold(self, inputs: torch.Tensor, remains: Optional[int]) -> torch.Tensor:
        """Recover the inputs, unfolding.
        Args:
            inputs: [torch.float32; [B, T // F, F x C]], folded tensor.
        Return:
            [torch.float32; [B, T, C]], recovered.
        """
        # B, _, F x C
        bsize, _, channels = inputs.shape
        # [B, T, C]
        recovered = inputs.reshape(bsize, -1, channels // self.factor)
        if remains is not None:
            # [B, T + R, C] -> [B, T, C]
            recovered = recovered[:, :-remains]
        return recovered
