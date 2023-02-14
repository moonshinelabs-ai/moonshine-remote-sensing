# TODO(nharada): Make this dataclasses instead for better type checking

model_params = {
    "resnet50_rgb": {
        "weight_url": "https://moonshine-prod-models.s3.us-west-2.amazonaws.com/moonshine/unet/unet50_fmow_3chan_step268380.pt",
        "encoder": "resnet50",
        "decoder": (256, 128, 64, 32, 32),
        "model_name": "UNet_ResNet50",
        "input_channels": 3,
        "output_channels": 32,
    }
}
