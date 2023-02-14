# Moonshine
Pretrained remote sensing models for the rest of us.

### What is Moonshine?
Moonshine is a software package that makes it easier to train models on remote sensing data like satellite imagery. Using Moonshine's pretrained foundation models, you can reduce the amount of labeled data required and reduce the training compute needed.

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
import pytorch.nn as nn
from moonshine.models.unet import UNet

def CloudSegmentation(nn.Module):
    def __init__(self):
        # Create a blank model based on the available architectures.
        self.backbone = UNet(name="resnet50_rgb")
        # Load both encoder and decoder weights. Some networks will want to not load the decoder.
        self.backbone.load_weights(encoder_weights="resnet50_rgb", decoder_weights="resnet50_rgb")
        # Run a per-pixel classifier on top of the output vectors.
        self.classifier = nn.Dense(2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return nn.softmax(x)
```

You can also configure data pre-processing to make sure your data is formatted the same way as the model pretraining was done.

```python
from moonshine.preprocessing import get_preprocessing_fn
preprocess_fn = get_preprocessing_fn(model="unet", data="fmow_rgb")
```

For more info and examples, [read the docs](). Using Moonshine for your academic project? Cite [this webpage]().

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