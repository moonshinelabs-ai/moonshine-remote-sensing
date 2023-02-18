##################
 Available Models
##################

While the list of available models is currently small, we intend to add
more over the coming months, especially with feedback from our users.
Training a single model is quite expensive, and as an open source
project our budget is small, so we want to make sure that we spend our
money wisely and create the most useful possible models.

Currently we support both pre-built models, as well as weights
pretrained on specific datasets. We will release more combinations over
time.

----
UNet
----

``resnet50_fmow_rgb``: A UNet with a ResNet-50 backbone that has been
pretrained on the `functional map of the world RGB
dataset <https://github.com/fMoW/dataset>`_. The model was trained using
masked autoencoding self-supervised learning, meaning that it should be
more task agnostic than a model pretrained on a specific target task.
