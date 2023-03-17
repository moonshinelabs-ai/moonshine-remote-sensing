"""The ABC for Moonshine models."""
import abc
import io
import logging
from json import decoder
from multiprocessing.sharedctypes import Value
from typing import Any, Optional

import torch
import torch.nn as nn
from smart_open import open
from torch.utils.model_zoo import load_url

from .logging import logger
from .model_parameters import model_params


class MoonshineModel(nn.Module, abc.ABC):
    """The base class of all Moonshine released models."""

    def __init__(self, name: str):
        """Create the moonshine base model.

        Args:
            name: A valid name for the architecture of this model.
        """
        super().__init__()

        self.name = name

    def _download_state_dict(self, location: str) -> dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "s3://" in location:
            with open(location, "rb") as f:
                buffer = io.BytesIO(f.read())
                return torch.load(buffer)
        else:
            return load_url(location, map_location=torch.device(device))

    def _load_state(
        self, encoder_weights: Optional[str], decoder_weights: Optional[str]
    ):
        new_dict = {}
        # Load the encoder weights
        if encoder_weights:
            state_dict = self._download_state_dict(encoder_weights)
            for k, v in state_dict.items():
                if "unet" in k and "encode" in k:
                    new_key = k.replace("model.", "")
                    new_dict[new_key] = v

        # Load the decoder weights
        if decoder_weights:
            state_dict = self._download_state_dict(decoder_weights)
            for k, v in state_dict.items():
                if "unet" in k and "decode" in k:
                    new_key = k.replace("model.", "")
                    new_dict[new_key] = v

        self.load_state_dict(new_dict, strict=False)

    def load_weights(
        self,
        encoder_weights: Optional[str],
        decoder_weights: Optional[str],
    ):
        """Load external weights for this model. Can be either a path or a
        named set.

        Args:
            encoder_weights: Either a path to .pt weights or a valid model name. If None, will not load encoder weights.
            decoder_weights: Either a path to .pt weights or a valid model name. If None, will not load decoder weights.
        """
        if not encoder_weights and not decoder_weights:
            raise ValueError(
                "Didn't get any weights at all, need either encoder or decoder."
            )

        encoder_url = encoder_weights
        if encoder_weights:
            if encoder_weights in model_params.keys():
                encoder_url = str(model_params[encoder_weights]["weight_url"])
            logger.info(f"Trying to load encoder weights from {encoder_url}")

        decoder_url = decoder_weights
        if decoder_weights:
            if decoder_weights in model_params.keys():
                decoder_url = str(model_params[decoder_weights]["weight_url"])
            logger.info(f"Trying to load decoder weights from {decoder_url}")

        self._load_state(encoder_url, decoder_url)

    def num_params(self):
        trainable_params = 0
        for p in self.parameters():
            if p.requires_grad:
                trainable_params += p.numel()

        return trainable_params

    def describe(self):
        description = f"Moonshine Model: {self.name}"
        description += f"   Number of trainable parameters: {self.num_params()}"

        return "\n".join(description)
