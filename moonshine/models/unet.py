import io
from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from loguru import logger
from ml_collections.config_dict import ConfigDict
from smart_open import open

from public.moonshine.moonshine.models.base import MoonshineModel

weight_mapping = {
    "fmow_rgb": "https://moonshine-prod-models.s3.us-west-2.amazonaws.com/moonshine/unet/unet50_fmow_3chan_step268380.pt"
}

class UNet50(MoonshineModel):
    def __init__(self, weights: Optional[str], input_channels: int = 12):
        super().__init__()

        self.model_name = "UNet_ResNet50"
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            decoder_channels=(256, 128, 64, 32, 32),
            in_channels=input_channels,
            classes=32,
        )

        # If passed, load pretrained weights for the backbone unet.
        if weights:
            if weights in weight_mapping.keys():
                logger.info(f"Got pretrained weights, going to load from moonshine")
                url = weight_mapping[weights]
                self.load_weights(url)
                logger.info(f"Loaded state dict for {weights}")
            elif ".pt" in weights:
                logger.info(f"Got custom weights, going to load from path {weights}")
                self.load_weights(weights)
                logger.info(f"Loaded state dict for {weights}")
            else:
                logger.error(f"Invalid weights, either specify a path or {weight_mapping.keys()}")
                raise Exception("Not a valid weights specification or path.")
        else:
            logger.info("No weights passed, starting from blank network.")


    def forward(self, x):
        return self.unet(x)
