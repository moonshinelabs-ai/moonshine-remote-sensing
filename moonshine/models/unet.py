from json import encoder

"""UNet models for segmentation and classification."""
from typing import Optional, Sequence

import segmentation_models_pytorch as smp

from .base import MoonshineModel
from .model_parameters import model_params


class UNet(MoonshineModel):
    """A basic UNet model, implemented under the hood with segmentation-models-pytorch. The model uses a ResNet backbone and skip connections, as in the original paper at https://arxiv.org/abs/1505.04597. Some parameters are fixed, since pre-trained weights require a consistant network structure."""

    def _build_model(self) -> smp.Unet:
        assert self.name in model_params.keys(), "Unsupported model type."

        encoder = model_params[self.name]["encoder"]
        channels = model_params[self.name]["input_channels"]
        decoder = model_params[self.name]["decoder"]
        classes = model_params[self.name]["output_channels"]

        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            decoder_channels=decoder,
            in_channels=channels,
            classes=classes,
        )

    def __init__(
        self,
        name,
    ):
        """Create the UNet, without loading the weights.

        Args:
            name: A valid name for the architecture of this model.
        """
        super().__init__(name=name)
        self.unet = self._build_model()

    def forward(self, x):
        """Run the forward pass of this model."""
        return self.unet(x)
