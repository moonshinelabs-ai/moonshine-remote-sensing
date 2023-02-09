# Moonshine
Pretrained remote sensing models for the rest of us.

### What is Moonshine?
Moonshine is a software package that makes it easier to train models on remote sensing data like satellite imagery. Using Moonshine's pretrained foundation models, you can reduce the amount of labeled data required for your tasks by 90% or more.

### Getting Started
The Moonshine Python package offers a light wrapper around our pretrained PyTorch models.

To start:

```
pip install moonshine
```

Then you can load the pretrained weights into your own model architecture and fine tune with your own data:

```
import pytorch.nn as nn
from moonshine.models.unet import UNet

def CloudSegmentation(nn.Module):
    def __init__(self):
        self.backbone = UNet(variety="resnet50", weights="fmow_rgb")
        self.classifier = nn.Dense(2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return nn.softmax(x)
```

For more info and examples, [read the docs](). Using Moonshine for your academic project? Cite [our paper]().

### FAQ
