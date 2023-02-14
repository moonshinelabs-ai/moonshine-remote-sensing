from json import encoder
from typing import Optional, Sequence

import segmentation_models_pytorch as smp

from .base import MoonshineModel
from .model_parameters import model_params


class UNet(MoonshineModel):
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
        super().__init__(name=name)
        self.unet = self._build_model()

    def forward(self, x):
        return self.unet(x)
