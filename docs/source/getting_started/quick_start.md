# Quick Start

## Installation

Install Moonshine via pip

```
$ pip install moonshine
```

or install from the latest Github code

```
$ pip install git+https://github.com/moonshinelabs-ai/moonshine
```

## Getting Started

The Moonshine Python package offers a light wrapper around our
pretrained PyTorch models. You can load the pretrained weights into your
own model architecture and fine tune with your own data:

```python
import torch.nn as nn
from moonshine.models.unet import UNet


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Create a blank model based on the available architectures.
        self.backbone = UNet(name="unet50_fmow_rgb")
        # If we are using pretrained weights, load them here. In
        # general, using the decoder weights isn't preferred unless
        # your downstream task is also a reconstruction task. We suggest
        # trying only the encoder first.
        self.backbone.load_weights(
            encoder_weights="unet50_fmow_rgb", decoder_weights=None
        )
        # Run a per-pixel classifier on top of the output vectors.
        self.classifier = nn.Conv2d(32, 2, (1, 1))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
```

You can also configure data pre-processing to make sure your data is
formatted the same way as the model pretraining was done.

```python
from moonshine.preprocessing import get_preprocessing_fn

preprocess_fn = get_preprocessing_fn(model="unet", dataset="fmow_rgb")
```
