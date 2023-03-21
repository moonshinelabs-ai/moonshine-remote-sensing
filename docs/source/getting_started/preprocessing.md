# Preprocessing

In order to maximize the benefits of the pretrained models, you must use the same input processing as the training process. The preprocessing functions vary between both expected type of data inputs, as well as model architectures.

## Model

| Model | Description                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------- |
| unet  | Formatting for the UNet series of architectures. These models expect multichannel inputs on a fixed basis. |

# Dataset

| Dataset                                                                                                                                                                                                                                | Description                                                                                                                                                                                                                                                                                                                                                    | Platforms                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| fmow_rgb                                                                                                                                                                                                                               | Applies mean/variance standardization to the RGB images from the functional map of the world dataset. Expects a channel format of RGB. Note that while statistics were computed on a specific dataset, we expect this formatting to work generally with RGB satellite images with roughly the distribution of the dataset (i.e. not too much water or clouds). | QuickBird-2, GeoEye-1, WorldView-2, WorldView3 |
| Applies mean/variance standardization to the multispectral 16-bit images from the functional map of the world dataset. Because some of the platforms used are only 4 channels, we train to be robust to zeros in the missing channels. | QuickBird-2, GeoEye-1, WorldView-2, WorldView3                                                                                                                                                                                                                                                                                                                 |                                                |
