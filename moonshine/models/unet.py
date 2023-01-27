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
            with open(weights, "rb") as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer)
                logger.info("State dict file opened, attempt parsing...")
            new_dict = {}
            for k, v in state_dict.items():
                if "unet" in k and "encode" in k:
                    new_key = k.replace("model.", "")
                    new_dict[new_key] = v
            self.load_state_dict(new_dict, strict=False)

            logger.info(f"Loaded state dict for {weights}")

    def forward(self, x):
        return self.unet(x)
