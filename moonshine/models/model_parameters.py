# TODO(nharada): Make this dataclasses instead for better type checking

model_params = {
    "unet50_fmow_rgb": {
        "weight_url": "https://moonshine-prod-models.s3.us-west-2.amazonaws.com/moonshine/unet/unet50_fmow_3chan_step268380.pt",
        "encoder": "resnet50",
        "decoder": (256, 128, 64, 32, 32),
        "input_channels": 3,
        "output_channels": 32,
    },
    "unet50_fmow_full": {
        "weight_url": "https://moonshine-prod-models.s3.us-west-2.amazonaws.com/moonshine/unet/unet50_fmow_allchan_step284000.pt",
        "encoder": "resnet50",
        "decoder": (256, 128, 64, 32, 32),
        "input_channels": 8,
        "output_channels": 32,
    },
    "unet50_sentinel2_l2a": {
        "weight_url": "https://moonshine-prod-models.s3.us-west-2.amazonaws.com/moonshine/unet/unet50_ben_allchan_step156500.pt",
        "encoder": "resnet50",
        "decoder": (256, 128, 64, 32, 32),
        "input_channels": 12,
        "output_channels": 32,
    },
}
