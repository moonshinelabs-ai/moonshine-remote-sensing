from typing import Optional

import segmentation_models_pytorch as smp
from loguru import logger

from .base import MoonshineModel

weight_mapping = {
    "fmow_rgb": "https://moonshine-prod-models.s3.us-west-2.amazonaws.com/moonshine/unet/unet50_fmow_3chan_step268380.pt"
}

variety_names = {
    "resnet50": "UNet_ResNet50",
}


class UNet(MoonshineModel):
    def _build_model(self) -> smp.Unet:
        if self.variety == "resnet50":
            return smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                decoder_channels=(256, 128, 64, 32, 32),
                in_channels=self.input_channels,
                classes=32,
            )

    def __init__(self, variety: str, weights: Optional[str], input_channels: int = 3):
        super().__init__()

        self.variety = variety
        self.input_channels = input_channels
        self.model_name = variety_names[variety]

        self.unet = self._build_model()

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
                logger.error(
                    f"Invalid weights, either specify a path or {weight_mapping.keys()}, got {weights}"
                )
                raise Exception("Not a valid weights specification or path.")
        else:
            logger.info("No weights passed, starting from blank network.")

    def forward(self, x):
        return self.unet(x)
