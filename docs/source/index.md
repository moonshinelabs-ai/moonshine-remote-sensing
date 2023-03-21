# Moonshine

**Moonshine** is a Python library that makes it easy for remote sensing
researchers, professionals, and enthusists to develop ML models on their
data. It provides [pre-trained models](getting_started/models.md) across a variety of datasets and
architectures, allowing you to reduce your labeling costs and compute
requirements for your own application.

![Pretrain your models to save time and compute | width=400](_static/images/pretrain_compare.png)

The above chart shows the difference between training the functional map
of the world classification task using our pre-trained model vs.
training from scratch. Training from scratch both performs worse
overall, and for the same level of accuracy we can train for 50% less
time. Check out the [quick start](getting_started/quick_start.md) section for further information,
including how to install the library.

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/quick_start.md
   getting_started/models.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples
   :glob:

   _examples/*

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 1
   :glob:

   api_reference/*

.. _Twitter: https://twitter.com/moonshineai
.. _Email: mailto:hello@moonshinelabs.ai
.. _Slack: https://join.slack.com/t/moonshinecommunity/shared_invite/zt-1rg1vnvmt-pleUR7TducaDiAhcmnqAQQ
```