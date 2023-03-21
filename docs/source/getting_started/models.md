# Available Models

While the list of available models is currently small, we intend to add
more over the coming months, especially with feedback from our users.
Training a single model is quite expensive, and as an open source
project our budget is small, so we want to make sure that we spend our
money wisely and create the most useful possible models.

Currently we support both pre-built models, as well as weights
pretrained on specific datasets. We will release more combinations over
time.

## UNet

### resnet50_fmow_rgb

**Architecture**: UNet\
**Backbone**: ResNet-50\
**Data Source**: QuickBird-2, GeoEye-1, WorldView-2, WorldView3\
**Data Format**: RGB

A UNet that has been pretrained on the [functional map of the world RGB
dataset](https://github.com/fMoW/dataset). The model was trained using
masked autoencoding self-supervised learning, meaning that it should be
more task agnostic than a model pretrained on a specific target task.

To pre-process data, use `fmow_rgb` mode. This mode expects input values
from 0..255 in RGB format.

### resnet50_fmow_full

**Architecture**: UNet\
**Backbone**: ResNet-50\
**Data Source**: QuickBird-2, GeoEye-1, WorldView-2, WorldView3\
**Data Format**: 4 or 8 channel multispectral

A UNet that has been pretrained on the [functional map of the world RGB
dataset](https://github.com/fMoW/dataset). The model was trained using
masked autoencoding self-supervised learning, meaning that it should be
more task agnostic than a model pretrained on a specific target task.
Compared to the `resnet50_fmow_rgb` dataset, this dataset uses the
multispectral inputs in either 4 or 8 channel format. The model has been
trained to deal with the latter 4 missing channels for images that do
not have 8 channels of data.

To pre-process data, use `fmow_full` mode. This mode expects input
values from 0..65535, as the default in the functional map of the world
dataset.
