import functools
from typing import Callable

import numpy as np

from .settings import DatasetSettings, ModelSettings


def get_dataset_settings(dataset: str) -> DatasetSettings:
    if dataset == "fmow_rgb":
        return DatasetSettings(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
        )
    else:
        raise ValueError("Invalid dataset type.")


def get_model_settings(model: str) -> ModelSettings:
    if model == "unet":
        return ModelSettings(colorspace="rgb")
    else:
        raise ValueError("Invalid dataset type.")


def _preprocess_fn(
    x: np.ndarray, model: ModelSettings, dataset: DatasetSettings
) -> np.ndarray:
    if model.colorspace == "bgr":
        x = x[..., ::-1].copy()

    if dataset.mean is not None:
        x = x - dataset.mean

    if dataset.std is not None:
        x = x / dataset.std

    return x


def get_preprocessing_fn(model: str, dataset: str) -> Callable:
    model_settings = get_model_settings(model)
    dataset_settings = get_dataset_settings(dataset)

    return functools.partial(
        _preprocess_fn, model=model_settings, dataset=dataset_settings
    )
