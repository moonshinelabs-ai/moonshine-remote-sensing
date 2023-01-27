import abc

import torch.nn as nn


class MoonshineModel(nn.Module, abc.ABC):
    """The base class of all Moonshine released models."""

    def __init__(self):
        super().__init__()

        self.model_name = "unnamed"

    def num_params(self):
        trainable_params = 0
        for p in self.parameters():
            if p.requires_grad:
                trainable_params += p.numel()

        return trainable_params

    def describe(self):
        description = f"Moonshine Model: {self.model_name}"
        description += f"   Number of trainable parameters: {self.num_params()}"

        return "\n".join(description)
