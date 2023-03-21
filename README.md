<br />
<p align="center">
    <a href="https://github.com/moonshinelabs-ai/moonshine">
      <img src="https://moonshine-assets.s3.us-west-2.amazonaws.com/theme_light_logo.png" width="50%"/>
    </a>
</p>

<h2><p align="center">Pretrained remote sensing models for the rest of us.</p></h2>

<h4><p align='center'>
<a href="https://moonshineai.readthedocs.io/en/latest/">[Read The Docs]</a>
- <a href="https://moonshineai.readthedocs.io/en/latest/getting_started/quick_start.html">[Quick Start]</a>
- <a href="http://www.moonshinelabs.ai/">[Website]</a>
</p></h4>

<p align="center">
    <a href="https://moonshineai.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://readthedocs.org/projects/moonshineai/badge/?version=latest">
    </a>
    <a href="https://pypi.org/project/moonshinelabs-ai/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/moonshine">
    </a>
    <a href="https://pypi.org/project/moonshinelabs-ai/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/moonshine">
    </a>
    <a href="https://pepy.tech/project/moonshine/">
        <img alt="PyPi Downloads" src="https://static.pepy.tech/personalized-badge/moonshine?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads/month">
    </a>
    <a href="https://join.slack.com/t/moonshinecommunity/shared_invite/zt-1rg1vnvmt-pleUR7TducaDiAhcmnqAQQ">
        <img alt="Chat on Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/moonshinelabs-ai/moonshine/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
    </a>
</p>
<br />

## What is Moonshine?

Moonshine is a Python package that makes it easier to train models on remote sensing data like satellite imagery. Using Moonshine's pretrained models, you can reduce the amount of labeled data required and reduce the training compute needed.

For more info and examples, [read the docs](https://moonshineai.readthedocs.io/en/latest).

## Installation

PyPI version:

```sh
pip install moonshine
```

Latest version from source:

```sh
pip install git+https://github.com/moonshinelabs-ai/moonshine
```

## Quick Start

The Moonshine Python package offers a light wrapper around our pretrained PyTorch models. You can load the pretrained weights into your own model architecture and fine tune with your own data:

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

You can also configure data pre-processing to make sure your data is formatted the same way as the model pretraining was done.

```python
from moonshine.preprocessing import get_preprocessing_fn
preprocess_fn = get_preprocessing_fn(model="unet", dataset="fmow_rgb")
```

## Citing

```
@misc{Harada:2023,
  Author = {Nate Harada},
  Title = {Moonshine},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/moonshinelabs-ai/moonshine}}
}
```

## License

This project is under MIT License.
