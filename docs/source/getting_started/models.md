# Available Models

While the list of available models is currently small, we intend to add more over the
coming months, especially with feedback from our users. Training a single model is quite
expensive, and as an open source project, our budget is small, so we want to make sure
that we spend our money wisely and create the most useful possible models.

Currently, we support both pre-built models, as well as weights pretrained on specific
datasets. We will release more combinations over time.

## UNet

### ResNet50 • FMOW RGB • [View Code](https://github.com/moonshinelabs-ai/moonshine/blob/main/moonshine/models/model_parameters.py#L4)

| Attribute    | Value                                                   |
| ------------ | ------------------------------------------------------- |
| Full Name    | `unet50_fmow_rgb`                                       |
| Architecture | UNet                                                    |
| Backbone     | ResNet-50                                               |
| Data Source  | QuickBird-2, GeoEye-1, WorldView-2, WorldView3          |
| Data Format  | RGB                                                     |
| Pretraining  | [Masked Autoencoding](https://arxiv.org/abs/2111.06377) |

A UNet that has been pretrained on the
[functional map of the world RGB dataset](https://github.com/fMoW/dataset). The model
was trained using masked autoencoding self-supervised learning, meaning that it should
be more task agnostic than a model pretrained on a specific target task.

To pre-process data, use `fmow_rgb` mode for the dataset and `unet` for the model. This
mode expects `uint8` input values from 0 - 255 in RGB ordering.

### ResNet50 • FMOW Multispectral • [View Code](https://github.com/moonshinelabs-ai/moonshine/blob/main/moonshine/models/model_parameters.py#L11)

| Attribute    | Value                                                   |
| ------------ | ------------------------------------------------------- |
| Full Name    | `unet50_fmow_full`                                      |
| Architecture | UNet                                                    |
| Backbone     | ResNet-50                                               |
| Data Source  | QuickBird-2, GeoEye-1, WorldView-2, WorldView3          |
| Data Format  | 4/8 channel multispectral                               |
| Pretraining  | [Masked Autoencoding](https://arxiv.org/abs/2111.06377) |

A UNet that has been pretrained on the
[functional map of the world RGB dataset](https://github.com/fMoW/dataset). The model
was trained using masked autoencoding self-supervised learning, meaning that it should
be more task agnostic than a model pretrained on a specific target task. Compared to the
`resnet50_fmow_rgb` dataset, this dataset uses the multispectral inputs in either 4 or 8
channel format. The model has been trained to deal with the latter 4 missing channels
for images that do not have 8 channels of data.

To pre-process data, use `fmow_full` mode for the dataset and `unet` for the model. This
mode expects `uint16` input values from 0 - 65535, as the default in the functional map
of the world dataset.

### ResNet50 • Sentinel-2 L2A • [View Code](https://github.com/moonshinelabs-ai/moonshine/blob/main/moonshine/models/model_parameters.py#L18)

| Attribute    | Value                                                   |
| ------------ | ------------------------------------------------------- |
| Full Name    | `unet50_sentinel2_l2a`                                  |
| Architecture | UNet                                                    |
| Backbone     | ResNet-50                                               |
| Data Source  | Sentinel 2 (L2A)                                        |
| Data Format  | 12 channel multispectral                                |
| Pretraining  | [Masked Autoencoding](https://arxiv.org/abs/2111.06377) |

A UNet that has been pretrained on the
[BigEarthNet dataset](https://bigearth.net/). The model
was trained using masked autoencoding self-supervised learning, meaning that it should
be more task agnostic than a model pretrained on a specific target task.

To pre-process data, use `sentinel-l2a` mode for the dataset and `unet` for the model. This
mode expects `uint16` input values from 0 - 65535, as the default for Sentinel-2 data.
