# Moonshine
Pretrained remote sensing models for the rest of us.

[![Documentation Status](https://readthedocs.org/projects/moonshineai/badge/?version=latest)](https://moonshineai.readthedocs.io/en/latest/?badge=latest)

### What is Moonshine?
Moonshine is a software package that makes it easier to train models on remote sensing data like satellite imagery. Using Moonshine's pretrained foundation models, you can reduce the amount of labeled data required and reduce the training compute needed.

For more info and examples, [read the docs](https://moonshineai.readthedocs.io/en/latest/?badge=latest).

### Installation
PyPI version:

```sh
pip install moonshine
```

Latest version from source:

```sh
pip install git+https://github.com/moonshinelabs-ai/moonshine
```

### Quick Start
The Moonshine Python package offers a light wrapper around our pretrained PyTorch models. You can load the pretrained weights into your own model architecture and fine tune with your own data:

```python
import torch.nn as nn
from moonshine.models.unet import UNet

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Create a blank model based on the available architectures.
        self.backbone = UNet(name="unet50_fmow_rgb")
        # Load both encoder and decoder weights. Some networks will want to not load the decoder.
        # To train from scratch just leave this off.
        self.backbone.load_weights(
            encoder_weights="unet50_fmow_rgb", decoder_weights="unet50_fmow_rgb"
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

### Citing

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

### License

This project is under MIT License.