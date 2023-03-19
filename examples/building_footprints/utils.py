import os

import geojson
import rasterio

import numpy as np

from shapely.geometry import shape, Point


def one_hot_3d(array, num_classes):
    # Get the shape of the input array
    shape = array.shape

    # Create an empty 3D array with the required shape
    one_hot_array = np.zeros((shape[0], shape[1], num_classes))

    # Fill the one-hot array
    for i in range(shape[0]):
        for j in range(shape[1]):
            class_index = array[i, j]
            one_hot_array[i, j, class_index] = 1

    return one_hot_array


def tile_to_rgb(tile):
    rgb = tile[[4, 2, 1], :, :]
    rgb[rgb >= 2000] = 2000
    rgb = rgb / 2000.0
    rgb = np.rollaxis(rgb, 0, 3)

    return rgb


def label_from_8band(path, label_type="geojson"):
    if label_type not in ("geojson", "mask"):
        raise

    # Get the label folder
    folder = os.path.dirname(path)
    up = "/".join(path.split("/")[0:-2])
    label_path = os.path.join(up, label_type)

    # Get the filename
    filename = os.path.basename(path)
    keyname = filename.split(".")[0]

    if label_type == "geojson":
        label_name = keyname.replace("8band_", "Geo_") + ".geojson"
    else:
        label_name = keyname.replace("8band_", "Mask_") + ".npy"

    # Combine them
    return os.path.join(label_path, label_name)


def mask_from_geojson(label, mask_shape, transform):
    M, N, C = mask_shape
    mask = np.zeros((M, N))
    if len(label.features) == 0:
        return mask

    rows, cols = np.meshgrid(np.arange(M), np.arange(N))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lats = np.stack(xs, axis=1)
    lons = np.stack(ys, axis=1)
    points = np.stack((lats, lons), axis=2)

    def fn(pt):
        point = Point(pt)
        for feature in label.features:
            polygon = shape(feature["geometry"])
            if polygon.contains(point):
                return 1
        return 0

    mask = np.apply_along_axis(fn, 2, points)
    return mask


def mask_from_tile(path, shape, transform):
    # Get the label
    label_path = label_from_8band(path)
    with open(label_path, "r") as f:
        label_data = geojson.load(f)

    # Turn the json label into another image
    return mask_from_geojson(label_data, shape, transform)


def read_tile(path):
    # Read the data
    with rasterio.open(path, "r") as f:
        data = f.read().astype(float)
        data = np.rollaxis(data, 0, 3)
        transform = f.transform

    return data, transform
