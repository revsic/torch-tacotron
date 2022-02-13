from typing import List

import torch


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Noam learning rate scheduler, introduced by Vaswani et al., 2017.
    """
    def __init__(self, optim: torch.optim.Optimizer, warmup: int, channels: int):
        self.warmup = warmup
        self.channels = channels
        super().__init__(optim)

    def get_lr(self) -> List[float]:
        """Compute learning rates, linear warmup and exponential decaying.
        """
        return [
            lr * self.channels ** -0.5 * min(
                self.last_epoch ** -0.5, self.last_epoch * self.warmup ** -1.5)
            for lr in self.base_lrs]
