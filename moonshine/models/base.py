import abc

import torch.nn as nn
from torch.utils.model_zoo import load_url


class MoonshineModel(nn.Module, abc.ABC):
    """The base class of all Moonshine released models."""

    def __init__(self):
        super().__init__()

        self.model_name = "unnamed"

    def load_weights(self, weights: str, load_decoder=True):
        state_dict = load_url(weights)
        new_dict = {}
        for k, v in state_dict.items():
            if "unet" in k and "encode" in k:
                new_key = k.replace("model.", "")
                new_dict[new_key] = v

            if load_decoder:
                if "decode" in k:
                    new_key = k.replace("model.", "")
                    new_dict[new_key] = v

        self.load_state_dict(new_dict, strict=False)

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
